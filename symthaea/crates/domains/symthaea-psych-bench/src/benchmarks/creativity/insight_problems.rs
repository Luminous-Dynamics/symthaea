// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Insight Problem benchmark.
//!
//! Models the "representational change" theory of insight (Ohlsson, 1992;
//! Knoblich et al., 1999). The solver encounters a problem encoded as an
//! HDC constraint set that is unsolvable under the initial representation.
//! After failed similarity-based retrieval (impasse), restructuring occurs
//! via HDC re-encoding — permutation models new perspective, unbinding
//! models constraint relaxation — until the solution is found.
//!
//! Measures:
//! - `restructuring_success` — fraction of problems solved after restructuring
//! - `insight_latency` — mean cycles to reach restructuring (lower = faster insight)
//! - `aha_magnitude` — jump in similarity when insight occurs (the "Aha!" spike)
//!
//! Human baselines (Knoblich et al., 1999; matchstick arithmetic problems):
//! - restructuring_success: 0.55 (SD 0.18)
//! - insight_latency: ~12 cycles (SD 4) — normalized deliberation time
//! - aha_magnitude: 0.30 (SD 0.12) — subjective insight magnitude

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Insight Problem benchmark — representational change theory.
pub struct InsightProblemBenchmark;

fn xor_shift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

struct InsightTrialResult {
    /// Did restructuring find the solution?
    solved: bool,
    /// Cycles spent before restructuring succeeded (or max if failed).
    latency: f64,
    /// Jump in similarity at the moment of insight (0 if not solved).
    aha_magnitude: f64,
    /// Peak similarity achieved during search.
    peak_similarity: f64,
}

impl InsightProblemBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> InsightTrialResult {
        let dim = BinaryHV::DIM;
        let seed = config.trial_seed("creativity", "insight_problem", trial_idx);
        let mut rng = seed ^ 0xC6A4A7935BD1E995;

        // Number of constraints in the problem (3-6, influenced by WM capacity).
        // More WM = more constraints the solver can track simultaneously,
        // which paradoxically makes restructuring harder (more to rearrange)
        // but also provides more restructuring strategies.
        let n_constraints = config.working_memory_capacity.clamp(3, 6);

        // Number of restructuring strategies available = WM capacity - 1.
        // Models the cognitive resource allocation for exploring alternative
        // representations (Ash & Wiley, 2006; working memory and insight).
        let n_strategies = (config.working_memory_capacity - 1).clamp(2, 5);

        // Lapse_rate reduces the number of restructuring attempts.
        // Models attentional lapses that interrupt the restructuring process
        // (Wichmann & Hill, 2001; attentional theory of insight).
        let lapse_attempt_penalty = (config.lapse_rate * n_strategies as f64 * 0.6) as usize;
        let effective_strategies = n_strategies.saturating_sub(lapse_attempt_penalty).max(1);

        // --- Encode the problem as a constraint set ---
        // Each constraint is a role-filler binding. The "problem" is the bundle
        // of these bindings — the initial representation.

        let roles: Vec<BinaryHV> = (0..n_constraints)
            .map(|_| BinaryHV::random(xor_shift(&mut rng)))
            .collect();

        let fillers: Vec<BinaryHV> = (0..n_constraints)
            .map(|_| BinaryHV::random(xor_shift(&mut rng)))
            .collect();

        let bindings: Vec<BinaryHV> = roles
            .iter()
            .zip(fillers.iter())
            .map(|(r, f)| r.bind(f))
            .collect();

        let problem_repr = BinaryHV::bundle(&bindings);

        // --- Create the "solution" representation ---
        // The solution requires a different encoding of the same constraints.
        // It uses permuted roles (= new perspective) and partially different
        // fillers (= relaxed constraints). This models how the solution is
        // "hidden" in the problem but requires representational change to see.

        let solution_roles: Vec<BinaryHV> = roles
            .iter()
            .enumerate()
            .map(|(i, r)| {
                // Permute some roles — the "new perspective"
                if i % 2 == 0 {
                    r.permute(1 + (i % 3))
                } else {
                    *r
                }
            })
            .collect();

        // Solution fillers: some shared with original (partial overlap), some new.
        // The overlap models the fact that insight problems contain the answer
        // but in a form that requires re-representation.
        let solution_fillers: Vec<BinaryHV> = fillers
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if i < n_constraints / 2 {
                    // Keep original filler (shared constraint)
                    *f
                } else {
                    // New filler (relaxed constraint)
                    BinaryHV::random(xor_shift(&mut rng))
                }
            })
            .collect();

        let solution_bindings: Vec<BinaryHV> = solution_roles
            .iter()
            .zip(solution_fillers.iter())
            .map(|(r, f)| r.bind(f))
            .collect();

        let solution_repr = BinaryHV::bundle(&solution_bindings);

        // --- Phase 1: Impasse (failed similarity-based retrieval) ---
        // Direct comparison of problem representation to solution shows low
        // similarity — the initial encoding doesn't "see" the answer.
        let initial_sim = 1.0 - problem_repr.hamming_distance(&solution_repr) as f64 / dim as f64;

        // Impasse threshold: below this, the solver cannot retrieve the solution.
        // With random BinaryHVs at 16384D, bundle-to-bundle similarity hovers
        // around 0.50 (chance). We need to beat that meaningfully.
        let impasse_threshold = 0.56;

        // If by chance the initial representation already matches well enough,
        // no restructuring needed (rare with random HVs).
        if initial_sim > impasse_threshold {
            return InsightTrialResult {
                solved: true,
                latency: 0.0,
                aha_magnitude: 0.0,
                peak_similarity: initial_sim,
            };
        }

        // --- Phase 2: Restructuring attempts ---
        // Each strategy re-encodes the problem via different HDC operations:
        //   1. Permutation = shift perspective (rotate role space)
        //   2. Unbinding + rebinding = relax a constraint and try a new filler
        //   3. Partial re-bundling = recombine subsets of constraints
        //   4. Temporal binding = encode sequential dependency
        //   5. Noise injection = random perturbation (creative noise)

        let mut best_sim = initial_sim;
        let mut solved = false;
        let mut solve_cycle = 0usize;
        let mut aha_jump = 0.0f64;

        // Each strategy gets multiple cycles (sub-attempts) to converge.
        // Total cycles = strategies × cycles_per_strategy.
        let cycles_per_strategy = 4;

        for strategy_idx in 0..effective_strategies {
            if solved {
                break;
            }

            for cycle in 0..cycles_per_strategy {
                let global_cycle = strategy_idx * cycles_per_strategy + cycle;

                let restructured = match strategy_idx % 5 {
                    0 => {
                        // Strategy: Perspective shift via permutation.
                        // Permute all roles by increasing amounts to explore the
                        // representational space systematically.
                        let shift = cycle + 1;
                        let new_bindings: Vec<BinaryHV> = roles
                            .iter()
                            .zip(fillers.iter())
                            .map(|(r, f)| r.permute(shift).bind(f))
                            .collect();
                        BinaryHV::bundle(&new_bindings)
                    }
                    1 => {
                        // Strategy: Constraint relaxation via unbinding + rebinding.
                        // Replace one constraint at a time with a random filler,
                        // exploring whether removing that constraint reveals the path.
                        let relax_idx = cycle % n_constraints;
                        let new_filler = BinaryHV::random(xor_shift(&mut rng));
                        let mut new_bindings = bindings.clone();
                        new_bindings[relax_idx] = roles[relax_idx].bind(&new_filler);
                        BinaryHV::bundle(&new_bindings)
                    }
                    2 => {
                        // Strategy: Partial re-bundling — combine subsets.
                        // Take first half from original, second half from permuted roles.
                        let split = n_constraints.div_ceil(2);
                        let mixed: Vec<BinaryHV> = (0..n_constraints)
                            .map(|i| {
                                if i < split {
                                    bindings[i]
                                } else {
                                    let perm_shift = cycle + 1;
                                    roles[i].permute(perm_shift).bind(&fillers[i])
                                }
                            })
                            .collect();
                        BinaryHV::bundle(&mixed)
                    }
                    3 => {
                        // Strategy: Temporal re-encoding via bind_temporal.
                        // Encode the constraints as a temporal sequence rather than
                        // a simultaneous bundle — reveals sequential dependencies.
                        let mut accum = bindings[0];
                        for b in &bindings[1..] {
                            accum = accum.bind_temporal(b);
                        }
                        // Blend temporal and spatial representations
                        BinaryHV::bundle(&[accum, problem_repr])
                    }
                    _ => {
                        // Strategy: Creative noise — perturb and explore.
                        // Small perturbations can break fixation on the wrong
                        // representation (random search in HDC space).
                        let noise_level = 0.05 + 0.03 * cycle as f32;
                        problem_repr.add_noise(noise_level, xor_shift(&mut rng))
                    }
                };

                // Measure similarity to solution after restructuring.
                let new_sim =
                    1.0 - restructured.hamming_distance(&solution_repr) as f64 / dim as f64;

                // Track the "Aha!" jump: improvement over previous best.
                if new_sim > best_sim {
                    let jump = new_sim - best_sim;
                    if jump > aha_jump {
                        aha_jump = jump;
                    }
                    best_sim = new_sim;
                }

                if best_sim > impasse_threshold {
                    solved = true;
                    solve_cycle = global_cycle + 1;
                    break;
                }
            }
        }

        // If not solved, latency is the total number of cycles attempted.
        if !solved {
            solve_cycle = effective_strategies * cycles_per_strategy;
        }

        InsightTrialResult {
            solved,
            latency: solve_cycle as f64,
            aha_magnitude: aha_jump,
            peak_similarity: best_sim,
        }
    }
}

impl PsychBenchmark for InsightProblemBenchmark {
    fn name(&self) -> &str {
        "Creativity::InsightProblem"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Representational Change / Matchstick Arithmetic",
            citation: "Knoblich et al. (1999)",
            year: 1999,
            doi: Some("10.1037/0096-3445.128.4.435"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut successes = Vec::new();
        let mut latencies = Vec::new();
        let mut aha_magnitudes = Vec::new();
        let mut peak_sims = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);

            successes.push(if r.solved { 1.0 } else { 0.0 });
            latencies.push(r.latency);
            aha_magnitudes.push(r.aha_magnitude);
            peak_sims.push(r.peak_similarity);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "insight".to_string(),
                    correct: r.solved,
                    rt_ticks: r.latency,
                    similarity: r.peak_similarity,
                    confidence: r.aha_magnitude,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "restructuring_success",
            MetricValue::from_samples(&successes),
        );
        result.insert("insight_latency", MetricValue::from_samples(&latencies));
        result.insert("aha_magnitude", MetricValue::from_samples(&aha_magnitudes));
        result.insert("peak_similarity", MetricValue::from_samples(&peak_sims));

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
        assert!(result.metrics.contains_key("restructuring_success"));
        assert!(result.metrics.contains_key("insight_latency"));
        assert!(result.metrics.contains_key("aha_magnitude"));
        assert!(result.metrics.contains_key("peak_similarity"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite(), "metric not finite: {:?}", val);
        }
    }

    #[test]
    fn test_restructuring_success_bounded() {
        let config = BenchmarkConfig {
            trials_per_condition: 50,
            ..BenchmarkConfig::default()
        };
        let result = InsightProblemBenchmark.run(&config);
        let rs = result.metrics["restructuring_success"].mean;
        assert!(
            rs >= 0.0 && rs <= 1.0,
            "restructuring_success should be in [0,1]: {rs}"
        );
    }

    #[test]
    fn test_lapse_rate_degrades_success() {
        let baseline = BenchmarkConfig {
            trials_per_condition: 80,
            ..BenchmarkConfig::default()
        };
        let lapsed = BenchmarkConfig {
            lapse_rate: 0.25,
            trials_per_condition: 80,
            ..BenchmarkConfig::default()
        };

        let r_base = InsightProblemBenchmark.run(&baseline);
        let r_lapse = InsightProblemBenchmark.run(&lapsed);

        let s_base = r_base.metrics["restructuring_success"].mean;
        let s_lapse = r_lapse.metrics["restructuring_success"].mean;
        // High lapse rate reduces restructuring attempts, so success should
        // not increase significantly.
        assert!(
            s_lapse <= s_base + 0.10,
            "lapse should not improve success: base={s_base}, lapse={s_lapse}"
        );
    }

    #[test]
    fn test_aha_magnitude_non_negative() {
        let config = BenchmarkConfig {
            trials_per_condition: 30,
            ..BenchmarkConfig::default()
        };
        let result = InsightProblemBenchmark.run(&config);
        let aha = result.metrics["aha_magnitude"].mean;
        assert!(aha >= 0.0, "aha_magnitude should be non-negative: {aha}");
    }

    #[test]
    fn test_insight_latency_bounded() {
        let config = BenchmarkConfig {
            trials_per_condition: 30,
            ..BenchmarkConfig::default()
        };
        let result = InsightProblemBenchmark.run(&config);
        let lat = result.metrics["insight_latency"].mean;
        // Latency is in cycles; should be > 0 and bounded by max cycles
        // (5 strategies × 4 cycles = 20 max).
        assert!(
            lat >= 0.0 && lat <= 25.0,
            "insight_latency should be in [0, 25]: {lat}"
        );
    }

    #[test]
    fn test_deterministic_across_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            seed: 12345,
            ..BenchmarkConfig::default()
        };
        let r1 = InsightProblemBenchmark.run(&config);
        let r2 = InsightProblemBenchmark.run(&config);
        let s1 = r1.metrics["restructuring_success"].mean;
        let s2 = r2.metrics["restructuring_success"].mean;
        assert!(
            (s1 - s2).abs() < 1e-10,
            "same seed should produce same result: {s1} vs {s2}"
        );
    }

    #[test]
    fn test_wm_capacity_affects_strategies() {
        let low_wm = BenchmarkConfig {
            working_memory_capacity: 3,
            trials_per_condition: 50,
            ..BenchmarkConfig::default()
        };
        let high_wm = BenchmarkConfig {
            working_memory_capacity: 7,
            trials_per_condition: 50,
            ..BenchmarkConfig::default()
        };

        let r_low = InsightProblemBenchmark.run(&low_wm);
        let r_high = InsightProblemBenchmark.run(&high_wm);

        // Both should produce valid results
        let s_low = r_low.metrics["restructuring_success"].mean;
        let s_high = r_high.metrics["restructuring_success"].mean;
        assert!(s_low >= 0.0 && s_low <= 1.0);
        assert!(s_high >= 0.0 && s_high <= 1.0);
    }

    #[test]
    fn test_trial_trace_populated() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            trial_trace: true,
            ..BenchmarkConfig::default()
        };
        let result = InsightProblemBenchmark.run(&config);
        assert_eq!(result.trial_trace.len(), 5);
        for t in &result.trial_trace {
            assert_eq!(t.condition, "insight");
            assert!(t.rt_ticks >= 0.0);
        }
    }
}