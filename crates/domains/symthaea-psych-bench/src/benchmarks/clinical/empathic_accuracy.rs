// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Empathic Accuracy Benchmark — Ickes (1993) paradigm.
//!
//! Tests the ability to infer emotional states from behavioral and verbal cues.
//! Each scenario presents a described situation with verbal and behavioral cues,
//! and the system must infer the actual emotional state of the target person.
//!
//! HDC implementation: Verbal cues, behavioral cues, and emotional states are
//! encoded as BinaryHV prototypes. Inference accuracy is measured as the
//! similarity between the inferred emotion HV (constructed from cue encoding)
//! and the actual emotion prototype. Scenarios span basic emotions, mixed
//! emotions, masked emotions, subtle cues, and cultural variation.
//!
//! Human baselines (Ickes, 1993; Ickes, Stinson, Bissonnette & Garcia, 1990):
//! - empathic_accuracy: r = 0.60 (SD ~ 0.15) — correlation between inferred
//!   and actual emotional states
//! - basic_accuracy: r = 0.75 (SD ~ 0.12) — basic unambiguous emotions
//! - mixed_accuracy: r = 0.50 (SD ~ 0.18) — mixed/ambivalent emotions
//! - masked_accuracy: r = 0.35 (SD ~ 0.20) — deliberately concealed emotions
//! - subtle_accuracy: r = 0.45 (SD ~ 0.16) — low-intensity or nuanced cues

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Empathic Accuracy benchmark implementing the Ickes (1993) paradigm.
pub struct EmpathicAccuracyBenchmark;

/// Category of empathic scenario, affecting difficulty and cue availability.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ScenarioCategory {
    /// Clear, unambiguous single emotion — easiest.
    Basic,
    /// Two or more simultaneous emotions — moderate difficulty.
    Mixed,
    /// Emotion deliberately concealed behind a social mask — hard.
    Masked,
    /// Low-intensity or culturally specific cues — hard.
    Subtle,
    /// Cross-cultural display rules alter expression — moderate-hard.
    Cultural,
}

/// A single empathic accuracy scenario.
struct EmpatheticScenario {
    /// Seed offset for deterministic HV generation.
    seed_offset: u64,
    /// Category of this scenario.
    category: ScenarioCategory,
    /// Number of verbal cues available (1-3).
    verbal_cue_count: usize,
    /// Number of behavioral cues available (1-4).
    behavioral_cue_count: usize,
    /// Emotional intensity (0.0 = subtle, 1.0 = intense).
    /// Controls how much noise is added to behavioral cues.
    intensity: f32,
    /// Signal-to-noise ratio for the true emotion in the cue bundle.
    /// Maps to `add_noise` flip probability: lower signal = more flips.
    signal_strength: f32,
    /// Number of distractor emotions blended in (0 for basic, 1-2 for mixed/masked).
    distractor_count: usize,
}

/// Build the 20 canonical scenarios spanning all categories.
fn build_scenarios() -> Vec<EmpatheticScenario> {
    vec![
        // ── Basic emotions (4 scenarios) ──
        EmpatheticScenario {
            seed_offset: 100,
            category: ScenarioCategory::Basic,
            verbal_cue_count: 2,
            behavioral_cue_count: 3,
            intensity: 0.85,
            signal_strength: 0.80,
            distractor_count: 0,
        },
        EmpatheticScenario {
            seed_offset: 101,
            category: ScenarioCategory::Basic,
            verbal_cue_count: 3,
            behavioral_cue_count: 2,
            intensity: 0.90,
            signal_strength: 0.75,
            distractor_count: 0,
        },
        EmpatheticScenario {
            seed_offset: 102,
            category: ScenarioCategory::Basic,
            verbal_cue_count: 1,
            behavioral_cue_count: 3,
            intensity: 0.70,
            signal_strength: 0.70,
            distractor_count: 0,
        },
        EmpatheticScenario {
            seed_offset: 103,
            category: ScenarioCategory::Basic,
            verbal_cue_count: 2,
            behavioral_cue_count: 2,
            intensity: 0.95,
            signal_strength: 0.85,
            distractor_count: 0,
        },
        // ── Mixed emotions (4 scenarios) ──
        EmpatheticScenario {
            seed_offset: 200,
            category: ScenarioCategory::Mixed,
            verbal_cue_count: 2,
            behavioral_cue_count: 3,
            intensity: 0.60,
            signal_strength: 0.50,
            distractor_count: 1,
        },
        EmpatheticScenario {
            seed_offset: 201,
            category: ScenarioCategory::Mixed,
            verbal_cue_count: 3,
            behavioral_cue_count: 2,
            intensity: 0.55,
            signal_strength: 0.45,
            distractor_count: 2,
        },
        EmpatheticScenario {
            seed_offset: 202,
            category: ScenarioCategory::Mixed,
            verbal_cue_count: 1,
            behavioral_cue_count: 3,
            intensity: 0.65,
            signal_strength: 0.55,
            distractor_count: 1,
        },
        EmpatheticScenario {
            seed_offset: 203,
            category: ScenarioCategory::Mixed,
            verbal_cue_count: 2,
            behavioral_cue_count: 2,
            intensity: 0.50,
            signal_strength: 0.40,
            distractor_count: 2,
        },
        // ── Masked emotions (4 scenarios) ──
        EmpatheticScenario {
            seed_offset: 300,
            category: ScenarioCategory::Masked,
            verbal_cue_count: 2,
            behavioral_cue_count: 2,
            intensity: 0.40,
            signal_strength: 0.30,
            distractor_count: 1,
        },
        EmpatheticScenario {
            seed_offset: 301,
            category: ScenarioCategory::Masked,
            verbal_cue_count: 1,
            behavioral_cue_count: 3,
            intensity: 0.35,
            signal_strength: 0.25,
            distractor_count: 1,
        },
        EmpatheticScenario {
            seed_offset: 302,
            category: ScenarioCategory::Masked,
            verbal_cue_count: 3,
            behavioral_cue_count: 1,
            intensity: 0.30,
            signal_strength: 0.20,
            distractor_count: 2,
        },
        EmpatheticScenario {
            seed_offset: 303,
            category: ScenarioCategory::Masked,
            verbal_cue_count: 2,
            behavioral_cue_count: 2,
            intensity: 0.45,
            signal_strength: 0.35,
            distractor_count: 1,
        },
        // ── Subtle cues (4 scenarios) ──
        EmpatheticScenario {
            seed_offset: 400,
            category: ScenarioCategory::Subtle,
            verbal_cue_count: 1,
            behavioral_cue_count: 2,
            intensity: 0.25,
            signal_strength: 0.35,
            distractor_count: 0,
        },
        EmpatheticScenario {
            seed_offset: 401,
            category: ScenarioCategory::Subtle,
            verbal_cue_count: 2,
            behavioral_cue_count: 1,
            intensity: 0.20,
            signal_strength: 0.30,
            distractor_count: 1,
        },
        EmpatheticScenario {
            seed_offset: 402,
            category: ScenarioCategory::Subtle,
            verbal_cue_count: 1,
            behavioral_cue_count: 3,
            intensity: 0.30,
            signal_strength: 0.40,
            distractor_count: 0,
        },
        EmpatheticScenario {
            seed_offset: 403,
            category: ScenarioCategory::Subtle,
            verbal_cue_count: 1,
            behavioral_cue_count: 2,
            intensity: 0.15,
            signal_strength: 0.25,
            distractor_count: 1,
        },
        // ── Cultural variation (4 scenarios) ──
        EmpatheticScenario {
            seed_offset: 500,
            category: ScenarioCategory::Cultural,
            verbal_cue_count: 2,
            behavioral_cue_count: 3,
            intensity: 0.55,
            signal_strength: 0.45,
            distractor_count: 1,
        },
        EmpatheticScenario {
            seed_offset: 501,
            category: ScenarioCategory::Cultural,
            verbal_cue_count: 1,
            behavioral_cue_count: 2,
            intensity: 0.60,
            signal_strength: 0.40,
            distractor_count: 1,
        },
        EmpatheticScenario {
            seed_offset: 502,
            category: ScenarioCategory::Cultural,
            verbal_cue_count: 3,
            behavioral_cue_count: 2,
            intensity: 0.50,
            signal_strength: 0.50,
            distractor_count: 0,
        },
        EmpatheticScenario {
            seed_offset: 503,
            category: ScenarioCategory::Cultural,
            verbal_cue_count: 2,
            behavioral_cue_count: 3,
            intensity: 0.45,
            signal_strength: 0.35,
            distractor_count: 2,
        },
    ]
}

/// Number of emotion prototypes in the affective space.
const NUM_EMOTION_PROTOTYPES: usize = 12;

/// XOR-shift PRNG step (deterministic).
fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

/// Results from a single trial (one pass over all 20 scenarios).
struct TrialResult {
    /// Overall empathic accuracy (mean similarity across all scenarios).
    empathic_accuracy: f64,
    /// Accuracy on basic emotion scenarios only.
    basic_accuracy: f64,
    /// Accuracy on mixed emotion scenarios only.
    mixed_accuracy: f64,
    /// Accuracy on masked emotion scenarios only.
    masked_accuracy: f64,
    /// Accuracy on subtle cue scenarios only.
    subtle_accuracy: f64,
    /// Accuracy on cultural variation scenarios only.
    cultural_accuracy: f64,
    /// Mean inference confidence (highest similarity among prototypes).
    mean_confidence: f64,
    /// Mean processing time proxy (ticks).
    mean_rt: f64,
}

impl EmpathicAccuracyBenchmark {
    /// Run a single trial across all 20 scenarios.
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        trace: &mut Vec<TrialOutcome>,
        global_trial_idx: &mut usize,
    ) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let seed = config.trial_seed("clinical", "empathic_accuracy", trial_idx);
        let mut rng = seed ^ 0xA3B1C4D5E6F70819;

        let scenarios = build_scenarios();

        // Generate emotion prototypes — quasi-orthogonal in HDC space
        // BinaryHV is fixed at 16,384 dimensions; seeds determine content.
        let emotion_prototypes: Vec<BinaryHV> = (0..NUM_EMOTION_PROTOTYPES)
            .map(|i| BinaryHV::random(seed.wrapping_add(1000 + i as u64 * 137)))
            .collect();

        // Difficulty and noise parameters
        let difficulty_attenuation = 1.0 - config.difficulty as f32 * 0.25;
        let time_pressure_noise = config.time_pressure as f32 * 0.10;
        let encoding_noise = config.effective_noise() as f32 * 0.2;
        // Social cognition provides a subtle boost, not a dominant signal.
        // Reduced from 0.12 to 0.04: the old value was a full 1.0 SD bonus
        // against baseline 0.60 ± 0.12, inflating z-scores.
        let social_bonus: f32 = if config.enable_social { 0.04 } else { 0.0 };

        // Per-category accumulators
        let mut basic_sims = Vec::new();
        let mut mixed_sims = Vec::new();
        let mut masked_sims = Vec::new();
        let mut subtle_sims = Vec::new();
        let mut cultural_sims = Vec::new();
        let mut all_sims = Vec::new();
        let mut all_confidences = Vec::new();
        let mut rt_sum = 0.0f64;

        for scenario in &scenarios {
            xor_shift(&mut rng);

            // Select target emotion
            let target_idx = (scenario.seed_offset as usize) % NUM_EMOTION_PROTOTYPES;
            let target_emotion = &emotion_prototypes[target_idx];

            // Effective signal: scenario signal * difficulty model * attenuation
            let effective_signal = scenario.signal_strength
                * diff_model.signal_multiplier(config.difficulty) as f32
                * difficulty_attenuation;

            // flip_probability for add_noise: lower signal → more flips
            // signal=1.0 → flip=0.0 (perfect copy), signal=0.0 → flip=0.5 (random)
            let base_flip = (1.0 - effective_signal) * 0.5;

            // Build cue HVs: each cue is the target emotion with noise added.
            // Verbal cues have higher fidelity; behavioral cues are modulated by intensity.
            let mut cue_hvs: Vec<BinaryHV> = Vec::new();

            // Verbal cues: language-mediated, slightly less noise
            for v in 0..scenario.verbal_cue_count {
                let cue_seed = seed
                    .wrapping_add(scenario.seed_offset)
                    .wrapping_add(v as u64 * 31);
                let verbal_flip = (base_flip * 0.85).clamp(0.0, 0.5);
                let cue = target_emotion.add_noise(verbal_flip, cue_seed);
                cue_hvs.push(cue);
            }

            // Behavioral cues: nonverbal, modulated by emotional intensity
            for b in 0..scenario.behavioral_cue_count {
                let cue_seed = seed
                    .wrapping_add(scenario.seed_offset)
                    .wrapping_add(2000 + b as u64 * 47);
                // Lower intensity → more noise on behavioral cues
                let intensity_factor = scenario.intensity;
                let behav_flip = (base_flip + (1.0 - intensity_factor) * 0.15).clamp(0.0, 0.5);
                let cue = target_emotion.add_noise(behav_flip, cue_seed);
                cue_hvs.push(cue);
            }

            // Add distractor emotion traces for mixed/masked scenarios.
            // Distractor noise scales with scenario difficulty: in hard scenarios
            // (masked/subtle), social masking produces ambiguous expression rather
            // than a clean alternative emotion (Ekman, 1992; Matsumoto et al., 2008).
            // Distractor flip is at least as high as the target cue's base flip,
            // ensuring distractors never outweigh genuine emotional signals.
            let distractor_flip = base_flip.max(0.25);
            for d in 0..scenario.distractor_count {
                let dist_idx = ((scenario.seed_offset as usize) + d + 3) % NUM_EMOTION_PROTOTYPES;
                let dist_seed = seed
                    .wrapping_add(scenario.seed_offset)
                    .wrapping_add(3000 + d as u64 * 61);
                let dist_cue = emotion_prototypes[dist_idx].add_noise(distractor_flip, dist_seed);
                cue_hvs.push(dist_cue);
            }

            // Aggregate all cues via majority-rule bundle
            let inferred = if cue_hvs.is_empty() {
                // No cues at all — random guess
                BinaryHV::random(rng)
            } else {
                BinaryHV::bundle(&cue_hvs)
            };

            // Measure similarity to target emotion
            let target_sim = inferred.similarity(target_emotion);

            // Find best-matching prototype (for confidence metric)
            let best_sim = emotion_prototypes
                .iter()
                .map(|p| inferred.similarity(p))
                .fold(f32::NEG_INFINITY, f32::max);

            // Category-dependent complexity noise: harder emotions are harder
            // to detect (mixed/masked emotions have more interference).
            let complexity_noise: f32 = match scenario.category {
                ScenarioCategory::Basic => 0.0,
                ScenarioCategory::Cultural => 0.04,
                ScenarioCategory::Mixed => 0.06,
                ScenarioCategory::Subtle => 0.08,
                ScenarioCategory::Masked => 0.10,
            };

            // Apply noise degradation, complexity penalty, and social bonus
            xor_shift(&mut rng);
            let noise_offset = (rng % 10000) as f32 / 10000.0 * time_pressure_noise;
            let adjusted_sim = (target_sim - encoding_noise - noise_offset - complexity_noise
                + social_bonus)
                .clamp(0.0, 1.0);

            all_sims.push(adjusted_sim as f64);
            all_confidences.push(best_sim as f64);

            match scenario.category {
                ScenarioCategory::Basic => basic_sims.push(adjusted_sim as f64),
                ScenarioCategory::Mixed => mixed_sims.push(adjusted_sim as f64),
                ScenarioCategory::Masked => masked_sims.push(adjusted_sim as f64),
                ScenarioCategory::Subtle => subtle_sims.push(adjusted_sim as f64),
                ScenarioCategory::Cultural => cultural_sims.push(adjusted_sim as f64),
            }

            // RT proxy: harder categories take longer
            let base_rt = match scenario.category {
                ScenarioCategory::Basic => 3.0,
                ScenarioCategory::Mixed => 5.0,
                ScenarioCategory::Masked => 6.0,
                ScenarioCategory::Subtle => 5.5,
                ScenarioCategory::Cultural => 4.5,
            };
            let tp_speedup = config.time_pressure * 1.5;
            let item_rt = (base_rt - tp_speedup).max(1.0);
            rt_sum += item_rt;

            if config.trial_trace {
                let cat_str = match scenario.category {
                    ScenarioCategory::Basic => "basic",
                    ScenarioCategory::Mixed => "mixed",
                    ScenarioCategory::Masked => "masked",
                    ScenarioCategory::Subtle => "subtle",
                    ScenarioCategory::Cultural => "cultural",
                };
                trace.push(TrialOutcome {
                    trial_idx: *global_trial_idx,
                    condition: cat_str.to_string(),
                    correct: adjusted_sim > 0.5,
                    rt_ticks: item_rt,
                    similarity: adjusted_sim as f64,
                    confidence: best_sim as f64,
                    response_idx: target_idx,
                    extra: BTreeMap::new(),
                });
                *global_trial_idx += 1;
            }
        }

        let mean = |v: &[f64]| -> f64 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };

        TrialResult {
            empathic_accuracy: mean(&all_sims),
            basic_accuracy: mean(&basic_sims),
            mixed_accuracy: mean(&mixed_sims),
            masked_accuracy: mean(&masked_sims),
            subtle_accuracy: mean(&subtle_sims),
            cultural_accuracy: mean(&cultural_sims),
            mean_confidence: mean(&all_confidences),
            mean_rt: rt_sum / scenarios.len() as f64,
        }
    }
}

impl PsychBenchmark for EmpathicAccuracyBenchmark {
    fn name(&self) -> &str {
        "Clinical::EmpathicAccuracy"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Empathic Accuracy (Ickes paradigm)",
            citation: "Ickes (1993)",
            year: 1993,
            doi: Some("10.1037/10153-000"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut empathic_accs = Vec::new();
        let mut basic_accs = Vec::new();
        let mut mixed_accs = Vec::new();
        let mut masked_accs = Vec::new();
        let mut subtle_accs = Vec::new();
        let mut cultural_accs = Vec::new();
        let mut confidences = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();
        let mut global_trial_idx = 0usize;

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial, &mut trace, &mut global_trial_idx);
            empathic_accs.push(r.empathic_accuracy);
            basic_accs.push(r.basic_accuracy);
            mixed_accs.push(r.mixed_accuracy);
            masked_accs.push(r.masked_accuracy);
            subtle_accs.push(r.subtle_accuracy);
            cultural_accs.push(r.cultural_accuracy);
            confidences.push(r.mean_confidence);
            rts.push(r.mean_rt);
        }

        result.insert(
            "empathic_accuracy",
            MetricValue::from_samples(&empathic_accs),
        );
        result.insert("basic_accuracy", MetricValue::from_samples(&basic_accs));
        result.insert("mixed_accuracy", MetricValue::from_samples(&mixed_accs));
        result.insert("masked_accuracy", MetricValue::from_samples(&masked_accs));
        result.insert("subtle_accuracy", MetricValue::from_samples(&subtle_accs));
        result.insert(
            "cultural_accuracy",
            MetricValue::from_samples(&cultural_accs),
        );
        result.insert("mean_confidence", MetricValue::from_samples(&confidences));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 5; // basic, mixed, masked, subtle, cultural
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

/// Run the empathic accuracy benchmark with default configuration.
pub fn run_empathic_accuracy_benchmark() -> BenchmarkResult {
    let config = BenchmarkConfig::default();
    EmpathicAccuracyBenchmark.run(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empathic_accuracy_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = EmpathicAccuracyBenchmark.run(&config);
        assert!(result.metrics.contains_key("empathic_accuracy"));
        assert!(result.metrics.contains_key("basic_accuracy"));
        assert!(result.metrics.contains_key("mixed_accuracy"));
        assert!(result.metrics.contains_key("masked_accuracy"));
        assert!(result.metrics.contains_key("subtle_accuracy"));
        assert!(result.metrics.contains_key("cultural_accuracy"));
        assert!(result.metrics.contains_key("mean_confidence"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = EmpathicAccuracyBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }

    #[test]
    fn test_empathic_accuracy_bounded() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = EmpathicAccuracyBenchmark.run(&config);
        let acc = result.metrics["empathic_accuracy"].mean;
        assert!(
            (0.0..=1.0).contains(&acc),
            "empathic_accuracy ({:.3}) out of [0, 1]",
            acc
        );
    }

    #[test]
    fn test_basic_easier_than_masked() {
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            ..Default::default()
        };
        let result = EmpathicAccuracyBenchmark.run(&config);
        let basic = result.metrics["basic_accuracy"].mean;
        let masked = result.metrics["masked_accuracy"].mean;
        assert!(
            basic > masked,
            "basic ({:.3}) should exceed masked ({:.3})",
            basic,
            masked
        );
    }

    #[test]
    fn test_category_ordering() {
        // Expected: basic > cultural/mixed > subtle/masked
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            ..Default::default()
        };
        let result = EmpathicAccuracyBenchmark.run(&config);
        let basic = result.metrics["basic_accuracy"].mean;
        let masked = result.metrics["masked_accuracy"].mean;
        let subtle = result.metrics["subtle_accuracy"].mean;
        // Basic should be the easiest
        assert!(
            basic > masked && basic > subtle,
            "basic ({:.3}) should be highest; masked={:.3}, subtle={:.3}",
            basic,
            masked,
            subtle
        );
    }

    #[test]
    fn test_deterministic_results() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            seed: 42,
            ..Default::default()
        };
        let r1 = EmpathicAccuracyBenchmark.run(&config);
        let r2 = EmpathicAccuracyBenchmark.run(&config);
        let m1 = r1.metrics["empathic_accuracy"].mean;
        let m2 = r2.metrics["empathic_accuracy"].mean;
        assert!(
            (m1 - m2).abs() < 1e-10,
            "same seed should produce identical results: {} vs {}",
            m1,
            m2
        );
    }

    #[test]
    fn test_time_pressure_reduces_rt() {
        let base = BenchmarkConfig {
            trials_per_condition: 5,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressed = BenchmarkConfig {
            time_pressure: 1.0,
            ..base.clone()
        };
        let r_base = EmpathicAccuracyBenchmark.run(&base);
        let r_press = EmpathicAccuracyBenchmark.run(&pressed);
        let rt_base = r_base.metrics["rt_ticks"].mean;
        let rt_press = r_press.metrics["rt_ticks"].mean;
        assert!(
            rt_press < rt_base,
            "time pressure should reduce RT: base={:.2}, pressed={:.2}",
            rt_base,
            rt_press
        );
    }

    #[test]
    fn test_social_bonus_improves_accuracy() {
        let without_social = BenchmarkConfig {
            trials_per_condition: 15,
            enable_social: false,
            ..Default::default()
        };
        let with_social = BenchmarkConfig {
            enable_social: true,
            ..without_social.clone()
        };
        let r_without = EmpathicAccuracyBenchmark.run(&without_social);
        let r_with = EmpathicAccuracyBenchmark.run(&with_social);
        let acc_without = r_without.metrics["empathic_accuracy"].mean;
        let acc_with = r_with.metrics["empathic_accuracy"].mean;
        assert!(
            acc_with >= acc_without,
            "social bonus should not decrease accuracy: without={:.3}, with={:.3}",
            acc_without,
            acc_with
        );
    }

    #[test]
    fn test_provenance() {
        let bench = EmpathicAccuracyBenchmark;
        let prov = bench.provenance().expect("should have provenance");
        assert_eq!(prov.year, 1993);
        assert!(prov.paradigm.contains("Ickes"));
        assert!(prov.citation.contains("Ickes"));
    }

    #[test]
    fn test_run_empathic_accuracy_benchmark_convenience() {
        let result = run_empathic_accuracy_benchmark();
        assert_eq!(result.benchmark, "Clinical::EmpathicAccuracy");
        assert!(result.metrics.contains_key("empathic_accuracy"));
    }

    #[test]
    fn test_trial_trace_populated() {
        let config = BenchmarkConfig {
            trials_per_condition: 2,
            trial_trace: true,
            ..Default::default()
        };
        let result = EmpathicAccuracyBenchmark.run(&config);
        // 20 scenarios * 2 trials = 40 trace entries
        assert_eq!(result.trial_trace.len(), 40);
        // Each entry has a valid category
        for entry in &result.trial_trace {
            assert!(
                ["basic", "mixed", "masked", "subtle", "cultural"]
                    .contains(&entry.condition.as_str()),
                "unexpected condition: {}",
                entry.condition
            );
        }
    }

    #[test]
    fn test_conditions_count() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = EmpathicAccuracyBenchmark.run(&config);
        assert_eq!(result.conditions, 5);
        assert_eq!(result.trials_per_condition, 3);
    }
}
