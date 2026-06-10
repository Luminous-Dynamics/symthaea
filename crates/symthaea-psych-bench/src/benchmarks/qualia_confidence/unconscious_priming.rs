// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unconscious Priming Benchmark — Sub/Supra-Threshold Dissociation
//!
//! Sub-threshold stimuli prime processing of related probes without GWT ignition.
//! Conscious and unconscious processing are qualitatively different (Dehaene 2006):
//! conscious primes produce stronger, longer-lasting effects than unconscious primes.
//!
//! ## Method
//!
//! 1. Create prime HV, related probe HV (similarity ~0.70 via noise), unrelated distractors
//! 2. For each of 8 activation levels (0.2 to 0.9):
//!    - Submit prime at that level with filler competition → process → check ignition
//!    - Compute broadcast-probe similarity (how much prime info entered the workspace)
//!    - Conscious primes produce broadcasts similar to probe → activation facilitation
//!    - Submit related probe (with facilitation boost) + distractor → measure winner
//!    - Separate into conscious (prime ignited) vs unconscious (no ignition) trials
//! 3. Compare priming magnitude across conditions
//!
//! ## Why facilitation, not shared workspace?
//!
//! GWT is stateless across cycles by design. In neuroscience, priming works by
//! pre-activating neural representations — a consciously processed prime triggers
//! global broadcasting that strongly pre-activates related representations, while
//! a subliminal prime only weakly pre-activates them. We model this as an activation
//! boost to the related probe proportional to the broadcast's similarity to it.
//!
//! ## Predictions (Dehaene et al. 2006)
//!
//! - Both conscious and unconscious primes should produce some priming
//! - Conscious priming effect > unconscious priming effect (Type I dissociation)
//! - There should be a clear ignition threshold
//!
//! Science:
//! - Dehaene, S. et al. (2006). Conscious, preconscious, and subliminal processing. Trends Cogn Sci.
//! - Marcel, A. J. (1983). Conscious and unconscious perception. Cognitive Psychology.

use crate::benchmarks::qualia_confidence::helpers::jitter_from_seed;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceConfig, WorkspaceContent};

/// Activation levels to test (0.2 to 0.9 in 0.1 steps).
const ACTIVATION_LEVELS: [f64; 8] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Trials per activation level.
const TRIALS_PER_LEVEL: usize = 50;

/// Base probe activation level (moderate — always above threshold).
const PROBE_ACTIVATION: f64 = 0.55;

/// Subliminal priming boost (small — Marcel 1983 shows weak but real effects).
const SUBLIMINAL_BOOST: f64 = 0.015;

/// Scaling factor for conscious facilitation (broadcast similarity → activation boost).
const FACILITATION_SCALE: f64 = 0.25;

/// Unconscious Priming Benchmark.
///
/// Tests that sub-threshold primes influence probe processing differently
/// from supra-threshold (conscious) primes, demonstrating the qualitative
/// distinction between conscious and unconscious processing.
pub struct UnconsciousPrimingBenchmark;

impl PsychBenchmark for UnconsciousPrimingBenchmark {
    fn name(&self) -> &str {
        "QualiaConfidence::UnconsciousPriming"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Conscious vs Unconscious Priming Dissociation",
            citation: "Dehaene et al. (2006); Marcel (1983)",
            year: 2006,
            doi: Some("10.1016/j.tics.2005.12.012"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        // Create prime and related probe (high similarity via noise addition)
        let prime_hv = BinaryHV::random(config.seed);
        // Related probe: add ~15% bit noise to prime → similarity ~0.70-0.85
        let related_probe = prime_hv.add_noise(0.15, config.seed.wrapping_add(100));

        let prime_probe_similarity = prime_hv.similarity(&related_probe) as f64;
        result.insert(
            "prime_probe_similarity",
            MetricValue::from_samples(&[prime_probe_similarity]),
        );

        // Track priming effects per condition
        let mut conscious_priming_effects = Vec::new();
        let mut unconscious_priming_effects = Vec::new();
        let mut ignition_counts = vec![0usize; ACTIVATION_LEVELS.len()];

        for (ai, &activation) in ACTIVATION_LEVELS.iter().enumerate() {
            let mut level_priming = Vec::new();

            for trial in 0..TRIALS_PER_LEVEL {
                let trial_seed = config
                    .seed
                    .wrapping_mul(0x100000001b3)
                    .wrapping_add(ai as u64 * 1000 + trial as u64);

                // Phase 1: Submit prime with filler competition
                // Filler content makes the workspace non-trivial — the prime must
                // compete for entry, creating a genuine ignition threshold.
                let prime_config = WorkspaceConfig {
                    entry_threshold: 0.50,
                    ..Default::default()
                };
                let mut prime_ws = GlobalWorkspace::new(prime_config);

                let jitter = jitter_from_seed(trial_seed, 0.03);
                let prime_activation = (activation + jitter).clamp(0.0, 1.0);

                let prime_content =
                    WorkspaceContent::new(vec![prime_hv], prime_activation, "prime".to_string());
                prime_ws.submit(prime_content);

                // Add filler domains that compete with the prime
                for f in 0..3u64 {
                    let filler_hv = BinaryHV::random(trial_seed.wrapping_add(200 + f));
                    let filler_content =
                        WorkspaceContent::new(vec![filler_hv], 0.45, format!("filler_{f}"));
                    prime_ws.submit(filler_content);
                }

                let prime_assessment = prime_ws.process();

                // Check if prime specifically entered conscious workspace
                let prime_in_conscious = prime_assessment
                    .conscious_contents
                    .iter()
                    .any(|c| c.source == "prime");
                let prime_ignited = prime_assessment.ignition_detected || prime_in_conscious;

                if prime_ignited {
                    ignition_counts[ai] += 1;
                }

                // Compute facilitation: how much the prime's processing pre-activates
                // the related probe's representation.
                //
                // Conscious primes get broadcast → broadcast HV is similar to prime →
                // prime is similar to probe → large facilitation.
                // Unconscious primes don't broadcast → only subliminal facilitation.
                let facilitation = if prime_in_conscious {
                    // Get broadcast similarity to the related probe
                    let broadcast_sim = if !prime_assessment.broadcasts.is_empty() {
                        prime_assessment.broadcasts[0].content[0].similarity(&related_probe) as f64
                    } else {
                        // Conscious but no broadcast — use prime-probe similarity
                        prime_hv.similarity(&related_probe) as f64
                    };
                    // Conscious facilitation: proportional to broadcast-probe overlap
                    let conscious_boost = (broadcast_sim - 0.5).max(0.0) * FACILITATION_SCALE;
                    conscious_boost + SUBLIMINAL_BOOST
                } else {
                    // Subliminal: weak but real (Marcel 1983)
                    SUBLIMINAL_BOOST
                };

                // Phase 2: Submit related probe (with facilitation) vs distractor
                let probe_config = WorkspaceConfig {
                    entry_threshold: 0.40,
                    ..Default::default()
                };
                let mut probe_ws = GlobalWorkspace::new(probe_config);

                let distractor_hv = BinaryHV::random(trial_seed.wrapping_add(500));

                // Probe gets facilitation boost from prime processing
                let boosted_probe_activation = (PROBE_ACTIVATION + facilitation).clamp(0.0, 1.0);

                let probe_content = WorkspaceContent::new(
                    vec![related_probe],
                    boosted_probe_activation,
                    "related_probe".to_string(),
                );
                // Distractor at base activation — no facilitation
                let distractor_content = WorkspaceContent::new(
                    vec![distractor_hv],
                    PROBE_ACTIVATION,
                    "distractor".to_string(),
                );

                probe_ws.submit(probe_content);
                probe_ws.submit(distractor_content);
                let probe_assessment = probe_ws.process();

                // Measure priming effect: did the probe win the competition?
                let probe_won = probe_assessment
                    .conscious_contents
                    .iter()
                    .any(|c| c.source == "related_probe");
                let distractor_won = probe_assessment
                    .conscious_contents
                    .iter()
                    .any(|c| c.source == "distractor");

                // Priming effect: +1 if probe wins, -1 if distractor, graded tie
                let priming_effect = if probe_won && !distractor_won {
                    1.0
                } else if distractor_won && !probe_won {
                    -1.0
                } else {
                    // Tie: use the facilitation magnitude as graded signal
                    facilitation.clamp(-1.0, 1.0)
                };

                level_priming.push(priming_effect);

                if prime_ignited {
                    conscious_priming_effects.push(priming_effect);
                } else {
                    unconscious_priming_effects.push(priming_effect);
                }
            }

            // Per-level metrics
            let level_mean = level_priming.iter().sum::<f64>() / level_priming.len() as f64;
            let act_pct = (activation * 100.0) as usize;
            result.insert(
                format!("priming_at_activation_{act_pct:03}"),
                MetricValue::from_samples(&[level_mean]),
            );
        }

        // Aggregate conscious vs unconscious priming
        let conscious_priming = if conscious_priming_effects.is_empty() {
            0.0
        } else {
            conscious_priming_effects.iter().sum::<f64>() / conscious_priming_effects.len() as f64
        };

        let unconscious_priming = if unconscious_priming_effects.is_empty() {
            0.0
        } else {
            unconscious_priming_effects.iter().sum::<f64>()
                / unconscious_priming_effects.len() as f64
        };

        result.insert(
            "conscious_priming_effect",
            MetricValue::from_samples(&[conscious_priming]),
        );
        result.insert(
            "unconscious_priming_effect",
            MetricValue::from_samples(&[unconscious_priming]),
        );
        result.insert(
            "priming_dissociation",
            MetricValue::from_samples(&[conscious_priming - unconscious_priming]),
        );

        // Type I dissociation: conscious > unconscious (binary)
        let type_i = if conscious_priming > unconscious_priming {
            1.0
        } else {
            0.0
        };
        result.insert("type_i_dissociation", MetricValue::from_samples(&[type_i]));

        // Ignition threshold: first activation level with >50% ignition rate
        let ignition_rates: Vec<f64> = ignition_counts
            .iter()
            .map(|&c| c as f64 / TRIALS_PER_LEVEL as f64)
            .collect();
        let ignition_threshold = ACTIVATION_LEVELS
            .iter()
            .zip(ignition_rates.iter())
            .find(|&(_, &rate)| rate > 0.5)
            .map(|(&level, _)| level)
            .unwrap_or(1.0);

        result.insert(
            "ignition_threshold",
            MetricValue::from_samples(&[ignition_threshold]),
        );

        // Report rate (fraction of all trials with ignition)
        let total_ignitions: usize = ignition_counts.iter().sum();
        let total_trials = ACTIVATION_LEVELS.len() * TRIALS_PER_LEVEL;
        let conscious_report_rate = total_ignitions as f64 / total_trials as f64;
        result.insert(
            "conscious_report_rate",
            MetricValue::from_samples(&[conscious_report_rate]),
        );

        // Per-level ignition rates
        for (ai, &activation) in ACTIVATION_LEVELS.iter().enumerate() {
            let act_pct = (activation * 100.0) as usize;
            result.insert(
                format!("ignition_rate_at_{act_pct:03}"),
                MetricValue::from_samples(&[ignition_rates[ai]]),
            );
        }

        result.conditions = ACTIVATION_LEVELS.len();
        result.trials_per_condition = TRIALS_PER_LEVEL;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            seed: 42,
            ..Default::default()
        }
    }

    #[test]
    fn test_unconscious_priming_runs() {
        let bench = UnconsciousPrimingBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "QualiaConfidence::UnconsciousPriming");
        assert!(result.metrics.len() >= 5);
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = UnconsciousPrimingBenchmark;
        let result = bench.run(&config());
        for (key, value) in &result.metrics {
            assert!(
                value.mean.is_finite(),
                "Metric '{key}' should be finite, got {}",
                value.mean
            );
        }
    }

    #[test]
    fn test_conscious_priming_exceeds_unconscious() {
        // Dehaene (2006): conscious primes produce stronger effects (Type I dissociation)
        let bench = UnconsciousPrimingBenchmark;
        let result = bench.run(&config());
        let conscious = result.metrics.get("conscious_priming_effect").unwrap().mean;
        let unconscious = result
            .metrics
            .get("unconscious_priming_effect")
            .unwrap()
            .mean;
        assert!(
            conscious > unconscious,
            "Conscious priming ({conscious}) should exceed unconscious ({unconscious})"
        );
    }

    #[test]
    fn test_type_i_dissociation() {
        // Binary dissociation marker: conscious > unconscious
        let bench = UnconsciousPrimingBenchmark;
        let result = bench.run(&config());
        let type_i = result.metrics.get("type_i_dissociation").unwrap().mean;
        assert!(
            type_i > 0.5,
            "Type I dissociation should be present: {type_i}"
        );
    }

    #[test]
    fn test_priming_dissociation_positive() {
        let bench = UnconsciousPrimingBenchmark;
        let result = bench.run(&config());
        let dissociation = result.metrics.get("priming_dissociation").unwrap().mean;
        assert!(
            dissociation > 0.01,
            "Priming dissociation (conscious - unconscious) should be positive: {dissociation}"
        );
    }

    #[test]
    fn test_ignition_threshold_reasonable() {
        let bench = UnconsciousPrimingBenchmark;
        let result = bench.run(&config());
        let threshold = result.metrics.get("ignition_threshold").unwrap().mean;
        assert!(
            threshold >= 0.2 && threshold <= 1.0,
            "Ignition threshold should be in valid range: {threshold}"
        );
    }

    #[test]
    fn test_prime_probe_similarity() {
        let bench = UnconsciousPrimingBenchmark;
        let result = bench.run(&config());
        let sim = result.metrics.get("prime_probe_similarity").unwrap().mean;
        assert!(
            sim > 0.6 && sim < 1.0,
            "Prime-probe similarity should be high but not perfect: {sim}"
        );
    }

    #[test]
    fn test_deterministic() {
        let bench = UnconsciousPrimingBenchmark;
        let r1 = bench.run(&config());
        let r2 = bench.run(&config());
        let d1 = r1.metrics.get("priming_dissociation").unwrap().mean;
        let d2 = r2.metrics.get("priming_dissociation").unwrap().mean;
        assert!(
            (d1 - d2).abs() < 1e-10,
            "Same seed should produce identical results: {d1} vs {d2}"
        );
    }

    #[test]
    fn test_ignition_increases_with_activation() {
        let bench = UnconsciousPrimingBenchmark;
        let result = bench.run(&config());
        let rate_low = result.metrics.get("ignition_rate_at_020").unwrap().mean;
        let rate_high = result.metrics.get("ignition_rate_at_090").unwrap().mean;
        assert!(
            rate_high >= rate_low,
            "Higher activation should produce more ignition: low={rate_low}, high={rate_high}"
        );
    }

    #[test]
    fn test_has_provenance() {
        let bench = UnconsciousPrimingBenchmark;
        assert!(bench.provenance().is_some());
    }

    #[test]
    fn test_dissociation_robust_across_seeds() {
        let bench = UnconsciousPrimingBenchmark;
        for seed in [42, 137, 256, 999, 7777] {
            let cfg = BenchmarkConfig {
                seed,
                ..Default::default()
            };
            let result = bench.run(&cfg);
            let conscious = result.metrics.get("conscious_priming_effect").unwrap().mean;
            let unconscious = result
                .metrics
                .get("unconscious_priming_effect")
                .unwrap()
                .mean;
            assert!(
                conscious >= unconscious,
                "Conscious priming should >= unconscious at seed={seed}: {conscious} vs {unconscious}"
            );
        }
    }
}
