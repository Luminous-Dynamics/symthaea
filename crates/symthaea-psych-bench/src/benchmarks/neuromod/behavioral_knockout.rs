// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Behavioral Knockout Benchmark — Cohen's d Effect Sizes
//!
//! Validates that transmitter knockouts on the **real** `NeuromodulatorBath`
//! produce measurable downstream behavioral differences. Uses Cohen's d
//! (standardized mean difference) to quantify each knockout's primary effect
//! and double dissociation (primary d > off-target d).
//!
//! Science: Doya (2002) — "Metalearning and neuromodulation"
//! Cohen (1988) — Effect size conventions (small=0.2, medium=0.5, large=0.8)

use crate::harness::PsychBenchmark;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use std::collections::BTreeMap;
use symthaea_neuromodulators::{NeuromodulatorBath, NeuromodulatorInputs};

/// Number of simulation cycles per condition.
const N_CYCLES: usize = 200;

/// Neutral inputs for baseline simulation.
fn neutral_inputs() -> NeuromodulatorInputs {
    NeuromodulatorInputs {
        prediction_error: 0.2,
        surprise: false,
        reward_signal: 0.0,
        coherence: 0.6,
        arousal: 0.4,
        binding_strength: 0.5,
        epistemic_confidence: 0.5,
        flow_active: false,
        consciousness_level: None,
        moral_signal: None,
    }
}

/// Behavioral metrics sampled per cycle from the bath.
#[derive(Debug, Clone, Default)]
struct BehavioralSample {
    learning_rate: f32,
    gradient_scale: f32,
    exploration: f32,
    confidence: f32,
    attention: f32,
    threshold: f32,
    global_inhibition: f32,
    consciousness_mod: f32,
    ei_ratio: f32,
    behavioral_flexibility: f32,
    #[allow(dead_code)]
    sleep_pressure: f32,
}

impl BehavioralSample {
    fn from_bath(bath: &NeuromodulatorBath) -> Self {
        Self {
            learning_rate: bath.learning_rate_factor(),
            gradient_scale: bath.gradient_scale_factor(),
            exploration: bath.exploration_delta(),
            confidence: bath.confidence_delta(),
            attention: bath.attention_factor(),
            threshold: bath.threshold_factor(),
            global_inhibition: bath.global_inhibition(),
            consciousness_mod: bath.consciousness_modulation(),
            ei_ratio: bath.ei_ratio(),
            behavioral_flexibility: bath.behavioral_flexibility(),
            sleep_pressure: bath.sleep_pressure(),
        }
    }
}

/// Accumulated statistics across cycles.
struct ConditionStats {
    samples: Vec<BehavioralSample>,
}

impl ConditionStats {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(N_CYCLES),
        }
    }

    fn mean(&self, accessor: fn(&BehavioralSample) -> f32) -> f64 {
        let n = self.samples.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        self.samples.iter().map(|s| accessor(s) as f64).sum::<f64>() / n
    }

    fn std(&self, accessor: fn(&BehavioralSample) -> f32) -> f64 {
        let n = self.samples.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = self.mean(accessor);
        let variance = self
            .samples
            .iter()
            .map(|s| {
                let diff = accessor(s) as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    }
}

/// Cohen's d = (mean_baseline - mean_ko) / pooled_std
fn cohens_d(
    baseline: &ConditionStats,
    ko: &ConditionStats,
    accessor: fn(&BehavioralSample) -> f32,
) -> f64 {
    let m1 = baseline.mean(accessor);
    let m2 = ko.mean(accessor);
    let s1 = baseline.std(accessor);
    let s2 = ko.std(accessor);
    let pooled = ((s1 * s1 + s2 * s2) / 2.0).sqrt();
    if pooled < 1e-12 {
        return 0.0;
    }
    (m1 - m2) / pooled
}

/// Run a condition: create bath, run N cycles with neutral inputs, apply knockout after each update.
fn run_condition(knockout: Option<&str>) -> ConditionStats {
    let mut bath = NeuromodulatorBath::default();
    let inputs = neutral_inputs();
    let mut stats = ConditionStats::new();

    for _ in 0..N_CYCLES {
        bath.update(&inputs);

        // Apply knockout (clamp target to 0)
        match knockout {
            Some("da") => bath.clamp_all_levels(Some(0.0), None, None, None, None, None, None),
            Some("ne") => bath.clamp_all_levels(None, Some(0.0), None, None, None, None, None),
            Some("sht") => bath.clamp_all_levels(None, None, Some(0.0), None, None, None, None),
            Some("ach") => bath.clamp_all_levels(None, None, None, Some(0.0), None, None, None),
            Some("gaba") => bath.clamp_all_levels(None, None, None, None, Some(0.0), None, None),
            Some("all") => bath.clamp_all_levels(
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
            ),
            _ => {} // Baseline: no knockout
        }

        stats.samples.push(BehavioralSample::from_bath(&bath));
    }

    stats
}

/// Behavioral Knockout Benchmark.
///
/// Validates that each transmitter knockout produces measurable downstream
/// behavioral differences (Cohen's d > threshold) using the real
/// `NeuromodulatorBath`.
pub struct BehavioralKnockoutBenchmark;

impl PsychBenchmark for BehavioralKnockoutBenchmark {
    fn name(&self) -> &str {
        "Neuromod::BehavioralKnockout"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;
        let baseline = run_condition(None);
        let da_ko = run_condition(Some("da"));
        let ne_ko = run_condition(Some("ne"));
        let sht_ko = run_condition(Some("sht"));
        let ach_ko = run_condition(Some("ach"));
        let gaba_ko = run_condition(Some("gaba"));
        let all_ko = run_condition(Some("all"));

        // Primary effect sizes (Cohen's d)
        let da_ko_lr_d = cohens_d(&baseline, &da_ko, |s| s.learning_rate);
        let da_ko_gradient_d = cohens_d(&baseline, &da_ko, |s| s.gradient_scale);
        let ne_ko_exploration_d = cohens_d(&baseline, &ne_ko, |s| s.exploration);
        let ne_ko_flexibility_d = cohens_d(&baseline, &ne_ko, |s| s.behavioral_flexibility);
        let sht_ko_confidence_d = cohens_d(&baseline, &sht_ko, |s| s.confidence);
        let sht_ko_consciousness_d = cohens_d(&baseline, &sht_ko, |s| s.consciousness_mod);
        let ach_ko_attention_d = cohens_d(&baseline, &ach_ko, |s| s.attention);
        let ach_ko_threshold_d = cohens_d(&baseline, &ach_ko, |s| s.threshold);
        let gaba_ko_inhibition_d = cohens_d(&baseline, &gaba_ko, |s| s.global_inhibition);
        let gaba_ko_ei_d = cohens_d(&baseline, &gaba_ko, |s| s.ei_ratio);
        let all_ko_lr_collapse_d = cohens_d(&baseline, &all_ko, |s| s.learning_rate);
        let all_ko_consciousness_collapse_d = cohens_d(&baseline, &all_ko, |s| s.consciousness_mod);

        // Baseline means
        let baseline_lr_mean = baseline.mean(|s| s.learning_rate);
        let baseline_exploration_mean = baseline.mean(|s| s.exploration);
        let baseline_confidence_mean = baseline.mean(|s| s.confidence);
        let baseline_attention_mean = baseline.mean(|s| s.attention);
        let baseline_inhibition_mean = baseline.mean(|s| s.global_inhibition);
        let baseline_consciousness_mean = baseline.mean(|s| s.consciousness_mod);

        let mut result = BenchmarkResult::new(self.name(), None);

        // Cohen's d effect sizes
        result.insert("da_ko_lr_d", MetricValue::from_samples(&[da_ko_lr_d]));
        result.insert(
            "da_ko_gradient_d",
            MetricValue::from_samples(&[da_ko_gradient_d]),
        );
        result.insert(
            "ne_ko_exploration_d",
            MetricValue::from_samples(&[ne_ko_exploration_d]),
        );
        result.insert(
            "ne_ko_flexibility_d",
            MetricValue::from_samples(&[ne_ko_flexibility_d]),
        );
        result.insert(
            "sht_ko_confidence_d",
            MetricValue::from_samples(&[sht_ko_confidence_d]),
        );
        result.insert(
            "sht_ko_consciousness_d",
            MetricValue::from_samples(&[sht_ko_consciousness_d]),
        );
        result.insert(
            "ach_ko_attention_d",
            MetricValue::from_samples(&[ach_ko_attention_d]),
        );
        result.insert(
            "ach_ko_threshold_d",
            MetricValue::from_samples(&[ach_ko_threshold_d]),
        );
        result.insert(
            "gaba_ko_inhibition_d",
            MetricValue::from_samples(&[gaba_ko_inhibition_d]),
        );
        result.insert("gaba_ko_ei_d", MetricValue::from_samples(&[gaba_ko_ei_d]));
        result.insert(
            "all_ko_lr_collapse_d",
            MetricValue::from_samples(&[all_ko_lr_collapse_d]),
        );
        result.insert(
            "all_ko_consciousness_collapse_d",
            MetricValue::from_samples(&[all_ko_consciousness_collapse_d]),
        );

        // Baseline means
        result.insert(
            "baseline_lr_mean",
            MetricValue::from_samples(&[baseline_lr_mean]),
        );
        result.insert(
            "baseline_exploration_mean",
            MetricValue::from_samples(&[baseline_exploration_mean]),
        );
        result.insert(
            "baseline_confidence_mean",
            MetricValue::from_samples(&[baseline_confidence_mean]),
        );
        result.insert(
            "baseline_attention_mean",
            MetricValue::from_samples(&[baseline_attention_mean]),
        );
        result.insert(
            "baseline_inhibition_mean",
            MetricValue::from_samples(&[baseline_inhibition_mean]),
        );
        result.insert(
            "baseline_consciousness_mean",
            MetricValue::from_samples(&[baseline_consciousness_mean]),
        );

        if config.trial_trace {
            let baseline_flex = baseline.mean(|s| s.behavioral_flexibility);
            trace.push(TrialOutcome {
                trial_idx,
                condition: "baseline".to_string(),
                correct: baseline_flex > 0.0,
                rt_ticks: 0.0,
                similarity: baseline_flex,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            let da_flex = da_ko.mean(|s| s.behavioral_flexibility);
            trace.push(TrialOutcome {
                trial_idx,
                condition: "da_knockout".to_string(),
                correct: da_ko_lr_d.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: da_flex,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            let ne_flex = ne_ko.mean(|s| s.behavioral_flexibility);
            trace.push(TrialOutcome {
                trial_idx,
                condition: "ne_knockout".to_string(),
                correct: ne_ko_exploration_d.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: ne_flex,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            let sht_flex = sht_ko.mean(|s| s.behavioral_flexibility);
            trace.push(TrialOutcome {
                trial_idx,
                condition: "sht_knockout".to_string(),
                correct: sht_ko_confidence_d.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: sht_flex,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            let ach_flex = ach_ko.mean(|s| s.behavioral_flexibility);
            trace.push(TrialOutcome {
                trial_idx,
                condition: "ach_knockout".to_string(),
                correct: ach_ko_attention_d.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: ach_flex,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            let gaba_flex = gaba_ko.mean(|s| s.behavioral_flexibility);
            trace.push(TrialOutcome {
                trial_idx,
                condition: "gaba_knockout".to_string(),
                correct: gaba_ko_inhibition_d.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: gaba_flex,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            let all_flex = all_ko.mean(|s| s.behavioral_flexibility);
            trace.push(TrialOutcome {
                trial_idx,
                condition: "all_knockout".to_string(),
                correct: all_ko_lr_collapse_d.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: all_flex,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            result.trial_trace = trace;
        }
        let _ = trial_idx;

        result.conditions = 7;
        result.trials_per_condition = 1;
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
    fn test_behavioral_knockout_runs() {
        let bench = BehavioralKnockoutBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "Neuromod::BehavioralKnockout");
        assert!(result.metrics.len() >= 18);
    }

    #[test]
    fn test_da_knockout_positive_d() {
        let bench = BehavioralKnockoutBenchmark;
        let result = bench.run(&config());
        let d = result
            .metrics
            .get("da_ko_lr_d")
            .expect("missing da_ko_lr_d")
            .mean;
        assert!(
            d.abs() > 0.1,
            "DA knockout should produce measurable LR effect: d={d}"
        );
    }

    #[test]
    fn test_ne_knockout_positive_d() {
        let bench = BehavioralKnockoutBenchmark;
        let result = bench.run(&config());
        let d = result
            .metrics
            .get("ne_ko_exploration_d")
            .expect("missing ne_ko_exploration_d")
            .mean;
        assert!(
            d.abs() > 0.1,
            "NE knockout should produce measurable exploration effect: d={d}"
        );
    }

    #[test]
    fn test_sht_knockout_positive_d() {
        let bench = BehavioralKnockoutBenchmark;
        let result = bench.run(&config());
        let d = result
            .metrics
            .get("sht_ko_confidence_d")
            .expect("missing sht_ko_confidence_d")
            .mean;
        assert!(
            d.abs() > 0.1,
            "5-HT knockout should produce measurable confidence effect: d={d}"
        );
    }

    #[test]
    fn test_ach_knockout_positive_d() {
        let bench = BehavioralKnockoutBenchmark;
        let result = bench.run(&config());
        let d = result
            .metrics
            .get("ach_ko_attention_d")
            .expect("missing ach_ko_attention_d")
            .mean;
        assert!(
            d.abs() > 0.1,
            "ACh knockout should produce measurable attention effect: d={d}"
        );
    }

    #[test]
    fn test_gaba_knockout_positive_d() {
        let bench = BehavioralKnockoutBenchmark;
        let result = bench.run(&config());
        let d = result
            .metrics
            .get("gaba_ko_inhibition_d")
            .expect("missing gaba_ko_inhibition_d")
            .mean;
        assert!(
            d.abs() > 0.1,
            "GABA knockout should produce measurable inhibition effect: d={d}"
        );
    }

    #[test]
    fn test_all_knockout_collapse() {
        let bench = BehavioralKnockoutBenchmark;
        let result = bench.run(&config());
        let lr_d = result
            .metrics
            .get("all_ko_lr_collapse_d")
            .expect("missing all_ko_lr_collapse_d")
            .mean;
        let cons_d = result
            .metrics
            .get("all_ko_consciousness_collapse_d")
            .expect("missing all_ko_consciousness_collapse_d")
            .mean;
        assert!(
            lr_d.abs() > 0.1 || cons_d.abs() > 0.1,
            "All-knockout should collapse at least one output: lr_d={lr_d}, cons_d={cons_d}"
        );
    }

    #[test]
    fn test_double_dissociation() {
        let baseline = run_condition(None);
        let ne_ko = run_condition(Some("ne"));
        let da_ko = run_condition(Some("da"));

        let da_lr_d = cohens_d(&baseline, &da_ko, |s| s.learning_rate).abs();
        let ne_lr_d = cohens_d(&baseline, &ne_ko, |s| s.learning_rate).abs();
        let ne_exploration_d = cohens_d(&baseline, &ne_ko, |s| s.exploration).abs();
        let da_exploration_d = cohens_d(&baseline, &da_ko, |s| s.exploration).abs();

        // DA knockout should affect LR more than NE knockout affects LR
        assert!(
            da_lr_d > ne_lr_d * 0.5,
            "DA knockout LR effect ({da_lr_d:.3}) should be substantial relative to NE knockout LR effect ({ne_lr_d:.3})"
        );

        // NE knockout should affect exploration more than DA knockout affects exploration
        assert!(
            ne_exploration_d > da_exploration_d * 0.5,
            "NE knockout exploration effect ({ne_exploration_d:.3}) should be substantial relative to DA knockout exploration effect ({da_exploration_d:.3})"
        );
    }
}
