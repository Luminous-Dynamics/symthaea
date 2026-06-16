// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Remote Associates Test (RAT).
//!
//! Given three cue words, find a fourth that connects them all.
//! Uses HDC bundling: encode each cue, bundle them, and check if the
//! solution word ranks highest among candidates by similarity.
//! Human baseline: ~0.50 accuracy (Bowden & Jung-Beeman 2003).

use crate::adapter::StimulusAdapter;
use crate::adapter::semantic::{RatTriadData, SemanticScenarioAdapter, Word};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Remote Associates Test benchmark.
pub struct RemoteAssociatesBenchmark;

struct RatTriad {
    cues: [&'static str; 3],
    solution: &'static str,
    distractors: [&'static str; 9],
}

impl RemoteAssociatesBenchmark {
    fn triads() -> Vec<RatTriad> {
        vec![
            RatTriad {
                cues: ["cottage", "swiss", "cake"],
                solution: "cheese",
                distractors: [
                    "bread", "milk", "house", "mountain", "sugar", "butter", "cream", "lake",
                    "flour",
                ],
            },
            RatTriad {
                cues: ["cream", "skate", "water"],
                solution: "ice",
                distractors: [
                    "snow", "cold", "lake", "rink", "fire", "glass", "stone", "wind", "rain",
                ],
            },
            RatTriad {
                cues: ["show", "life", "row"],
                solution: "boat",
                distractors: [
                    "car", "ship", "train", "road", "wave", "sail", "dock", "fish", "ocean",
                ],
            },
            RatTriad {
                cues: ["night", "wrist", "stop"],
                solution: "watch",
                distractors: [
                    "clock", "time", "guard", "tower", "light", "band", "hand", "bell", "ring",
                ],
            },
            RatTriad {
                cues: ["rocking", "wheel", "high"],
                solution: "chair",
                distractors: [
                    "table", "seat", "bench", "stool", "desk", "sofa", "throne", "swing", "stand",
                ],
            },
            RatTriad {
                cues: ["home", "sea", "bed"],
                solution: "sick",
                distractors: [
                    "room", "sleep", "rest", "shore", "wave", "dream", "night", "sand", "warm",
                ],
            },
            RatTriad {
                cues: ["man", "glove", "life"],
                solution: "love",
                distractors: [
                    "hand", "heart", "soul", "ring", "game", "fight", "care", "hope", "hate",
                ],
            },
            RatTriad {
                cues: ["board", "magic", "death"],
                solution: "black",
                distractors: [
                    "white", "dark", "night", "card", "game", "trick", "spell", "ghost", "skull",
                ],
            },
            RatTriad {
                cues: ["fish", "mine", "rush"],
                solution: "gold",
                distractors: [
                    "silver", "copper", "iron", "coal", "river", "pan", "dig", "cave", "ore",
                ],
            },
            RatTriad {
                cues: ["measure", "worm", "video"],
                solution: "tape",
                distractors: [
                    "film", "record", "disc", "reel", "wire", "glue", "string", "band", "roll",
                ],
            },
            RatTriad {
                cues: ["cross", "rain", "tie"],
                solution: "bow",
                distractors: [
                    "knot", "string", "arrow", "cloud", "wind", "silk", "loop", "ribbon", "lace",
                ],
            },
            RatTriad {
                cues: ["dream", "break", "light"],
                solution: "day",
                distractors: [
                    "night", "sun", "moon", "star", "dawn", "dusk", "time", "dark", "glow",
                ],
            },
            RatTriad {
                cues: ["print", "berry", "bird"],
                solution: "blue",
                distractors: [
                    "red", "green", "black", "white", "gold", "ink", "wing", "nest", "tree",
                ],
            },
            RatTriad {
                cues: ["pine", "crab", "sauce"],
                solution: "apple",
                distractors: [
                    "orange", "grape", "lemon", "fruit", "peach", "cherry", "plum", "mango",
                    "berry",
                ],
            },
            RatTriad {
                cues: ["base", "snow", "dance"],
                solution: "ball",
                distractors: [
                    "game", "play", "round", "field", "court", "bat", "goal", "net", "team",
                ],
            },
        ]
    }

    fn build_triad_data() -> Vec<RatTriadData> {
        Self::triads()
            .into_iter()
            .map(|t| RatTriadData {
                cues: [
                    t.cues[0].to_string(),
                    t.cues[1].to_string(),
                    t.cues[2].to_string(),
                ],
                solution: t.solution.to_string(),
                distractors: t.distractors.iter().map(|d| d.to_string()).collect(),
            })
            .collect()
    }

    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        adapter: &SemanticScenarioAdapter,
    ) -> (f64, f64, f64, f64) {
        let dim = config.dimension;
        let triads = Self::triads();
        let triad = &triads[trial_idx % triads.len()];

        // Encode cues and bundle them
        let cue_hvs: Vec<ContinuousHV> = triad
            .cues
            .iter()
            .map(|c| adapter.encode(&Word(c.to_string()), dim))
            .collect();
        let bundle = ContinuousHV::bundle_owned(&cue_hvs);

        // Encode solution and distractors
        let solution_hv = adapter.encode(&Word(triad.solution.to_string()), dim);
        let distractor_hvs: Vec<ContinuousHV> = triad
            .distractors
            .iter()
            .map(|d| adapter.encode(&Word(d.to_string()), dim))
            .collect();

        // Score all candidates by similarity to bundle.
        // Encoding noise degrades associative retrieval — higher noise reduces
        // discriminability between solution and distractors (Mednick, 1962 RAT;
        // individual differences in spreading activation noise).
        let enc_noise = config.effective_noise() as f32;
        // Time pressure: 0.08/unit noise disrupts similarity ranking, modeling reduced search
        // depth in associative retrieval under deadline (Mednick, 1962 RAT; Luce, 1986).
        let pressure_noise = config.time_pressure as f32 * 0.08;
        let seed = config.trial_seed("creativity", "rat", trial_idx);
        let sol_noise = {
            let ns = seed.wrapping_add(7000);
            ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let solution_sim = bundle.similarity(&solution_hv) + sol_noise * enc_noise * 0.20;
        let mut all_sims: Vec<(usize, f32)> = vec![(0, solution_sim)]; // index 0 = solution
        for (i, dhv) in distractor_hvs.iter().enumerate() {
            // Per-candidate encoding noise (hash-based deterministic)
            let cand_noise = {
                let ns = seed.wrapping_add(7100 + i as u64 * 13);
                ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            };
            // Alternating noise sign disrupts ranking under pressure
            let noise = if i % 2 == 0 {
                pressure_noise
            } else {
                -pressure_noise
            };
            all_sims.push((
                i + 1,
                bundle.similarity(dhv) + noise + cand_noise * enc_noise * 0.20,
            ));
        }
        all_sims.sort_by(|(_, a), (_, b)| b.total_cmp(a));

        // Accuracy: solution ranks first. Lapse model can flip correctness.
        let rank = all_sims.iter().position(|(idx, _)| *idx == 0).unwrap_or(9) + 1;
        let correct = config.check_correct(rank == 1, "remote_associates", trial_idx);
        let accuracy = if correct { 1.0 } else { 0.0 };
        let mean_rank = rank as f64;

        // Binding-based associative recall: convergent binding ensemble.
        // The RAT requires convergent association — finding the common link
        // across all three cues (Mednick, 1962). We compute all three
        // pairwise bindings and bundle them into a "convergence" vector,
        // then combine with the straight bundle for robust retrieval.
        //
        // The pairwise bindings capture multiplicative interactions between
        // cue pairs (structural co-occurrence), while the bundle captures
        // additive overlap (shared semantic links). Combining both implements
        // convergent retrieval where the answer must relate to multiple cues
        // simultaneously (Kanerva, 2009: distributed representations).
        let bind_01 = cue_hvs[0].bind(&cue_hvs[1]);
        let bind_02 = cue_hvs[0].bind(&cue_hvs[2]);
        let bind_12 = cue_hvs[1].bind(&cue_hvs[2]);
        let convergence = ContinuousHV::bundle(&[&bind_01, &bind_02, &bind_12]);

        let sol_ensemble =
            0.6 * bundle.similarity(&solution_hv) + 0.4 * convergence.similarity(&solution_hv);

        let best_dist_ensemble = distractor_hvs
            .iter()
            .map(|d| 0.6 * bundle.similarity(d) + 0.4 * convergence.similarity(d))
            .fold(f32::NEG_INFINITY, f32::max);
        let binding_accuracy = if sol_ensemble > best_dist_ensemble {
            1.0
        } else {
            0.0
        };

        // RT proxy: gap between solution and best distractor similarity
        let best_distractor_sim = all_sims
            .iter()
            .filter(|(idx, _)| *idx != 0)
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let margin = (solution_sim - best_distractor_sim).abs() as f64;
        let base = 4.0;
        let range = 7.0;
        let rt = base + (1.0 - margin.min(1.0)) * range;

        (accuracy, mean_rank, binding_accuracy, rt)
    }
}

impl PsychBenchmark for RemoteAssociatesBenchmark {
    fn name(&self) -> &str {
        "Creativity::RemoteAssociates"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Remote Associates Test",
            citation: "Mednick (1962)",
            year: 1962,
            doi: Some("10.1037/h0048850"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let triad_data = Self::build_triad_data();
        let adapter = SemanticScenarioAdapter::for_rat(&triad_data, config.dimension, config.seed);

        let mut accuracies = Vec::new();
        let mut ranks = Vec::new();
        let mut binding_accs = Vec::new();
        let mut rt_ticks = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (acc, rank, bind_acc, rt) = self.run_trial(config, trial, &adapter);
            accuracies.push(acc);
            ranks.push(rank);
            binding_accs.push(bind_acc);
            rt_ticks.push(rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "remote_associates".to_string(),
                    correct: acc > 0.5,
                    rt_ticks: rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("overall_accuracy", MetricValue::from_samples(&accuracies));
        result.insert("mean_solution_rank", MetricValue::from_samples(&ranks));
        result.insert(
            "convergent_binding",
            MetricValue::from_samples(&binding_accs),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rt_ticks));

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
    fn test_remote_associates_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 256,
            ..Default::default()
        };
        let result = RemoteAssociatesBenchmark.run(&config);
        assert!(result.metrics.contains_key("overall_accuracy"));
        assert!(result.metrics.contains_key("mean_solution_rank"));
        assert!(result.metrics.contains_key("convergent_binding"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite());
        }
    }

    #[test]
    fn test_remote_associates_values() {
        let config = BenchmarkConfig::default();
        let result = RemoteAssociatesBenchmark.run(&config);
        for (key, val) in &result.metrics {
            eprintln!("RAT {key}: mean={:.4}, sd={:.4}", val.mean, val.std_dev);
        }
    }
}
