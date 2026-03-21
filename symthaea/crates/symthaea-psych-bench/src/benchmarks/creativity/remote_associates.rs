//! Remote Associates Test (RAT).
//!
//! Given three cue words, find a fourth that connects them all.
//! Uses HDC bundling: encode each cue, bundle them, and check if the
//! solution word ranks highest among candidates by similarity.
//! Human baseline: ~0.50 accuracy (Bowden & Jung-Beeman 2003).

use crate::adapter::semantic::{RatTriadData, SemanticScenarioAdapter, Word};
use crate::adapter::StimulusAdapter;
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
        // Multi-level associative bundling: combine direct cue representations
        // with pairwise bindings. The bindings capture compound associations
        // (e.g., "cottage-swiss" → cheese) that the flat bundle misses.
        // Kenett et al. (2014, Creativity Research Journal) showed that creative
        // associative thinking involves both direct and mediated associations
        // through network spreading activation (Mednick, 1962).
        let pair_01 = cue_hvs[0].bind(&cue_hvs[1]);
        let pair_02 = cue_hvs[0].bind(&cue_hvs[2]);
        let pair_12 = cue_hvs[1].bind(&cue_hvs[2]);
        let raw_bundle = ContinuousHV::weighted_bundle(
            &[
                &cue_hvs[0],
                &cue_hvs[1],
                &cue_hvs[2],
                &pair_01,
                &pair_02,
                &pair_12,
            ],
            &[1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
        );

        // Lapse_rate degrades associative binding coherence: higher lapse → noisier
        // bundle representation, modeling reduced spreading activation depth
        // (Mednick 1962; individual differences in associative search breadth).
        let bundle = if config.lapse_rate > 0.0 {
            let corruption = (config.lapse_rate * 1.6) as f32; // up to 40% binding disruption
            let noise_hv =
                ContinuousHV::random(dim, config.trial_seed("creativity", "rat_lapse", trial_idx));
            ContinuousHV::weighted_bundle(
                &[&raw_bundle, &noise_hv],
                &[1.0 - corruption, corruption],
            )
        } else {
            raw_bundle
        };

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
        // Lapse_rate amplifies encoding noise, modeling reduced associative search
        // depth and noisier semantic representations (Mednick, 1962; Beaty et al., 2014).
        let lapse_noise_boost = config.lapse_rate as f32 * 0.15;
        let enc_noise = enc_noise + lapse_noise_boost;
        // Time pressure: 0.08/unit noise disrupts similarity ranking, modeling reduced search
        // depth in associative retrieval under deadline (Mednick, 1962 RAT; Luce, 1986).
        let pressure_noise = config.time_pressure as f32 * 0.08;
        let seed = config.trial_seed("creativity", "rat", trial_idx);
        let sol_noise = {
            let ns = seed.wrapping_add(7000);
            ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        // Reduced noise coefficient from 0.20 to 0.14: the bundled cue representation
        // provides a stronger convergent signal toward the solution than previously
        // modeled. In Mednick's (1962) associative hierarchy theory, the solution
        // word sits at the intersection of three activation gradients; HDC bundling
        // naturally computes this intersection with cleaner signal separation.
        let solution_sim = bundle.similarity(&solution_hv) + sol_noise * enc_noise * 0.14;
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
                bundle.similarity(dhv) + noise + cand_noise * enc_noise * 0.14,
            ));
        }
        // Lapse_rate controls associative search depth: higher lapse → fewer
        // candidates considered before committing to a response, modeling
        // premature search termination (Mednick, 1962; reduced spreading
        // activation breadth). This creates stable per-subject differences.
        if config.lapse_rate > 0.0 && all_sims.len() > 2 {
            let search_cutoff = {
                let hash = seed
                    .wrapping_mul(0x517CC1B727220A95)
                    .wrapping_add(trial_idx as u64 * 31);
                (hash >> 32) as f64 / (1u64 << 32) as f64
            };
            // Higher lapse → higher chance of dropping last candidate(s)
            let drop_prob = config.lapse_rate * 2.0; // up to 50% chance at max lapse
            if search_cutoff < drop_prob {
                // Remove last candidate from consideration (wasn't "found")
                all_sims.truncate(all_sims.len() - 1);
            }
        }

        all_sims.sort_by(|(_, a), (_, b)| b.total_cmp(a));

        // Accuracy: solution ranks first. Lapse model can flip correctness.
        let rank = all_sims
            .iter()
            .position(|(idx, _)| *idx == 0)
            .unwrap_or(all_sims.len())
            + 1;
        let correct = config.check_correct(rank == 1, "remote_associates", trial_idx);
        let accuracy = if correct { 1.0 } else { 0.0 };
        // Fractional rank: integer rank + similarity-margin interpolation for
        // continuous individual differences (rank alone is integer 1-4, too coarse
        // for reliable ICC). Margin captures how close the solution was to adjacent
        // candidates, reflecting associative search depth.
        let sol_sim_val = all_sims
            .iter()
            .find(|(idx, _)| *idx == 0)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let rank_pos = rank - 1; // 0-indexed position in sorted list
        let fractional_offset = if rank_pos > 0 {
            // How far solution's similarity is below the candidate just above it
            let above_sim = all_sims[rank_pos - 1].1;
            let gap = (above_sim - sol_sim_val).max(0.0).min(0.1);
            gap as f64 * 5.0 // scale to [0, 0.5]
        } else {
            // Rank 1: how far above the next candidate
            let below_sim = if all_sims.len() > 1 {
                all_sims[1].1
            } else {
                0.0
            };
            let margin = (sol_sim_val - below_sim).max(0.0).min(0.1);
            -(margin as f64 * 5.0) // negative offset = better than rank 1.0
        };
        let mean_rank = (rank as f64 + fractional_offset).max(0.5);

        // Binding-based associative recall: cue1.bind(cue2) similarity to solution
        let binding = cue_hvs[0].bind(&cue_hvs[1]);
        let binding_sim = binding.similarity(&solution_hv);
        let max_distractor_bind_sim = distractor_hvs
            .iter()
            .map(|d| binding.similarity(d))
            .fold(0.0f32, f32::max);
        let binding_accuracy = if binding_sim > max_distractor_bind_sim {
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
        result.insert("binding_accuracy", MetricValue::from_samples(&binding_accs));
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
        assert!(result.metrics.contains_key("binding_accuracy"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite());
        }
    }
}
