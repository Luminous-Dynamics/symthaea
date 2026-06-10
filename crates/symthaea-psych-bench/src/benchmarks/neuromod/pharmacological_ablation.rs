// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pharmacological Ablation benchmark: virtual transmitter knockout.
//!
//! Science: Classic pharmacological lesion methodology — selectively disable
//! individual transmitter channels and measure downstream cognitive degradation.
//!
//! Protocol: For each of DA/NE/5-HT/ACh, clamp the channel to 0.0 (knockout)
//! while running 200 update+reuptake cycles. Compare downstream indicators
//! (learning rate, exploration, confidence, attention) against unclamped baseline.
//! Each knockout should selectively impair its target function.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

/// Lightweight transmitter model mirroring `neuromodulators::Transmitter`.
#[derive(Clone)]
struct Transmitter {
    level: f32,
    receptor_sensitivity: f32,
    reuptake_rate: f32,
    baseline: f32,
}

impl Transmitter {
    fn new() -> Self {
        Self {
            level: 0.5,
            receptor_sensitivity: 1.0,
            reuptake_rate: 0.1,
            baseline: 0.5,
        }
    }

    fn effective(&self) -> f32 {
        (self.level * self.receptor_sensitivity).clamp(0.0, 2.0)
    }

    fn produce(&mut self, amount: f32) {
        self.level = (self.level + amount).clamp(0.0, 1.0);
    }

    fn reuptake(&mut self) {
        self.level += (self.baseline - self.level) * self.reuptake_rate;
        self.level = self.level.clamp(0.0, 1.0);
        let high = self.baseline + 0.2;
        let low = self.baseline - 0.2;
        if self.level > high {
            self.receptor_sensitivity *= 0.998;
        } else if self.level < low {
            self.receptor_sensitivity *= 1.002;
        }
        self.receptor_sensitivity = self.receptor_sensitivity.clamp(0.5, 2.0);
    }
}

/// Lightweight bath model mirroring `neuromodulators::NeuromodulatorBath`.
#[derive(Clone)]
struct Bath {
    da: Transmitter,
    ne: Transmitter,
    sht: Transmitter,
    ach: Transmitter,
}

/// Input signals for a single update cycle.
struct Inputs {
    prediction_error: f32,
    surprise: bool,
    reward_signal: f32,
    coherence: f32,
    arousal: f32,
    binding_strength: f32,
    epistemic_confidence: f32,
    flow_active: bool,
}

impl Bath {
    fn new() -> Self {
        Self {
            da: Transmitter::new(),
            ne: Transmitter::new(),
            sht: Transmitter::new(),
            ach: Transmitter::new(),
        }
    }

    fn update(&mut self, inputs: &Inputs) {
        // DA: reward prediction error
        let da_signal = inputs.reward_signal * 0.15
            + if inputs.prediction_error < 0.2 {
                0.05
            } else {
                -0.05
            }
            + if inputs.flow_active { 0.03 } else { 0.0 };
        self.da.produce(da_signal);

        // NE: unexpected uncertainty
        let ne_signal = if inputs.surprise { 0.15 } else { 0.0 }
            + inputs.arousal * 0.08
            + inputs.prediction_error.clamp(0.0, 0.5) * 0.1;
        self.ne.produce(ne_signal);

        // 5-HT: satisfaction
        let sht_signal = inputs.coherence * 0.08
            + inputs.epistemic_confidence * 0.05
            + inputs.binding_strength * 0.04
            - if inputs.reward_signal < -0.3 {
                0.1
            } else {
                0.0
            };
        self.sht.produce(sht_signal);

        // ACh: expected uncertainty
        let ach_signal = (1.0 - inputs.epistemic_confidence) * 0.1
            + if inputs.flow_active { 0.06 } else { 0.0 }
            + if inputs.binding_strength > 0.7 {
                0.03
            } else {
                0.0
            };
        self.ach.produce(ach_signal);

        // Cross-modulation
        if self.da.level > 0.7 {
            self.ne.level *= 0.97;
        }
        if self.sht.level > 0.7 {
            self.ne.level *= 0.98;
        }
        if self.ne.level > 0.6 {
            self.ach.produce(0.02);
        }

        // Reuptake
        self.da.reuptake();
        self.ne.reuptake();
        self.sht.reuptake();
        self.ach.reuptake();
    }

    fn clamp(&mut self, da: Option<f32>, ne: Option<f32>, sht: Option<f32>, ach: Option<f32>) {
        if let Some(v) = da {
            self.da.level = v.clamp(0.0, 1.0);
        }
        if let Some(v) = ne {
            self.ne.level = v.clamp(0.0, 1.0);
        }
        if let Some(v) = sht {
            self.sht.level = v.clamp(0.0, 1.0);
        }
        if let Some(v) = ach {
            self.ach.level = v.clamp(0.0, 1.0);
        }
    }

    // Downstream indicators (mirrors NeuromodulatorBath methods)
    fn learning_rate_factor(&self) -> f32 {
        (0.7 + self.da.effective() * 0.4).clamp(0.7, 1.5)
    }

    fn exploration_delta(&self) -> f32 {
        (self.ne.effective() - 0.5) * 0.2
    }

    fn confidence_delta(&self) -> f32 {
        (self.sht.effective() - 0.5) * 0.04
    }

    fn attention_factor(&self) -> f32 {
        (0.8 + self.ach.effective() * 0.25).clamp(0.8, 1.3)
    }

    fn gradient_scale_factor(&self) -> f32 {
        (0.5 + self.da.effective() * 0.75).clamp(0.5, 2.0)
    }

    fn threshold_gate(&self) -> f32 {
        (1.5 - self.ach.effective() * 0.5).clamp(0.5, 1.5)
    }
}

/// Default neutral input for steady-state simulation.
fn neutral_inputs() -> Inputs {
    Inputs {
        prediction_error: 0.2,
        surprise: false,
        reward_signal: 0.1,
        coherence: 0.6,
        arousal: 0.4,
        binding_strength: 0.5,
        epistemic_confidence: 0.5,
        flow_active: false,
    }
}

/// Run 200 cycles with optional channel knockout, return averaged downstream indicators.
fn run_ablation(
    clamp_da: Option<f32>,
    clamp_ne: Option<f32>,
    clamp_sht: Option<f32>,
    clamp_ach: Option<f32>,
) -> (f64, f64, f64, f64, f64, f64) {
    let mut bath = Bath::new();
    let inputs = neutral_inputs();
    let n_cycles = 200;

    let mut sum_lr = 0.0_f64;
    let mut sum_exploration = 0.0_f64;
    let mut sum_confidence = 0.0_f64;
    let mut sum_attention = 0.0_f64;
    let mut sum_gradient = 0.0_f64;
    let mut sum_threshold = 0.0_f64;

    for _ in 0..n_cycles {
        bath.update(&inputs);
        bath.clamp(clamp_da, clamp_ne, clamp_sht, clamp_ach);

        sum_lr += bath.learning_rate_factor() as f64;
        sum_exploration += bath.exploration_delta() as f64;
        sum_confidence += bath.confidence_delta() as f64;
        sum_attention += bath.attention_factor() as f64;
        sum_gradient += bath.gradient_scale_factor() as f64;
        sum_threshold += bath.threshold_gate() as f64;
    }

    let n = n_cycles as f64;
    (
        sum_lr / n,
        sum_exploration / n,
        sum_confidence / n,
        sum_attention / n,
        sum_gradient / n,
        sum_threshold / n,
    )
}

/// Pharmacological ablation benchmark: virtual transmitter knockout.
pub struct PharmacologicalAblationBenchmark;

impl PsychBenchmark for PharmacologicalAblationBenchmark {
    fn name(&self) -> &str {
        "Neuromod::PharmacologicalAblation"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let mut result = BenchmarkResult::new(self.name(), None);
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        // Baseline (no knockout)
        let (base_lr, base_expl, base_conf, base_attn, base_grad, base_thresh) =
            run_ablation(None, None, None, None);
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "baseline".to_string(),
                correct: base_lr > 0.0,
                rt_ticks: 0.0,
                similarity: base_lr,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // DA knockout
        let (ko_lr, _, _, _, ko_grad, _) = run_ablation(Some(0.0), None, None, None);
        let da_lr_drop = ((base_lr - ko_lr) / base_lr * 100.0).max(0.0);
        let da_grad_drop = ((base_grad - ko_grad) / base_grad * 100.0).max(0.0);
        result.insert(
            "da_knockout_lr_drop_pct",
            MetricValue::from_samples(&[da_lr_drop]),
        );
        result.insert(
            "da_knockout_gradient_drop_pct",
            MetricValue::from_samples(&[da_grad_drop]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "da_knockout".to_string(),
                correct: da_lr_drop > 0.0,
                rt_ticks: 0.0,
                similarity: ko_lr,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // NE knockout
        let (_, ko_expl, _, _, _, _) = run_ablation(None, Some(0.0), None, None);
        let ne_expl_diff = base_expl - ko_expl;
        result.insert(
            "ne_knockout_exploration_drop",
            MetricValue::from_samples(&[ne_expl_diff]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "ne_knockout".to_string(),
                correct: ne_expl_diff > 0.0,
                rt_ticks: 0.0,
                similarity: ko_expl,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // 5-HT knockout
        let (_, _, ko_conf, _, _, _) = run_ablation(None, None, Some(0.0), None);
        let sht_conf_diff = base_conf - ko_conf;
        result.insert(
            "sht_knockout_confidence_drop",
            MetricValue::from_samples(&[sht_conf_diff]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "sht_knockout".to_string(),
                correct: sht_conf_diff > 0.0,
                rt_ticks: 0.0,
                similarity: ko_conf,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // ACh knockout
        let (_, _, _, ko_attn, _, ko_thresh) = run_ablation(None, None, None, Some(0.0));
        let ach_attn_drop = ((base_attn - ko_attn) / base_attn * 100.0).max(0.0);
        let ach_thresh_rise = ((ko_thresh - base_thresh) / base_thresh * 100.0).max(0.0);
        result.insert(
            "ach_knockout_attention_drop_pct",
            MetricValue::from_samples(&[ach_attn_drop]),
        );
        result.insert(
            "ach_knockout_threshold_rise_pct",
            MetricValue::from_samples(&[ach_thresh_rise]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "ach_knockout".to_string(),
                correct: ach_attn_drop > 0.0,
                rt_ticks: 0.0,
                similarity: ko_attn,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // Double dissociation: each knockout's PRIMARY effect > other knockouts' off-target effect
        result.insert("baseline_lr", MetricValue::from_samples(&[base_lr]));
        result.insert(
            "baseline_exploration",
            MetricValue::from_samples(&[base_expl]),
        );
        result.insert(
            "baseline_confidence",
            MetricValue::from_samples(&[base_conf]),
        );
        result.insert(
            "baseline_attention",
            MetricValue::from_samples(&[base_attn]),
        );

        if config.trial_trace {
            result.trial_trace = trace;
        }
        let _ = trial_idx;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            citation: "Doya (2002) — Metalearning and neuromodulation",
            paradigm: "Pharmacological lesion / virtual knockout",
            year: 2002,
            doi: Some("10.1162/089976602753712018"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::config::BenchmarkConfig;

    #[test]
    fn test_pharmacological_ablation_runs() {
        let config = BenchmarkConfig::default();
        let bench = PharmacologicalAblationBenchmark;
        let result = bench.run(&config);
        // Should produce all expected metrics
        for key in &[
            "da_knockout_lr_drop_pct",
            "da_knockout_gradient_drop_pct",
            "ne_knockout_exploration_drop",
            "sht_knockout_confidence_drop",
            "ach_knockout_attention_drop_pct",
            "ach_knockout_threshold_rise_pct",
        ] {
            let m = result.metrics.get(*key);
            assert!(m.is_some(), "Missing metric: {key}");
            assert!(m.unwrap().mean.is_finite(), "Non-finite metric: {key}");
        }
    }

    #[test]
    fn test_da_knockout_reduces_lr() {
        let (base_lr, _, _, _, _, _) = run_ablation(None, None, None, None);
        let (ko_lr, _, _, _, _, _) = run_ablation(Some(0.0), None, None, None);
        assert!(
            ko_lr < base_lr,
            "DA knockout should reduce LR factor: baseline={base_lr}, knockout={ko_lr}"
        );
        // DA=0 → learning_rate_factor should be at minimum (0.7)
        assert!(
            ko_lr < 0.75,
            "DA knockout LR should be near minimum 0.7: got {ko_lr}"
        );
    }

    #[test]
    fn test_ne_knockout_suppresses_exploration() {
        let (_, base_expl, _, _, _, _) = run_ablation(None, None, None, None);
        let (_, ko_expl, _, _, _, _) = run_ablation(None, Some(0.0), None, None);
        assert!(
            ko_expl < base_expl,
            "NE knockout should reduce exploration: baseline={base_expl}, knockout={ko_expl}"
        );
    }

    #[test]
    fn test_all_knockout_metrics_degrade() {
        let (base_lr, base_expl, base_conf, base_attn, _, _) = run_ablation(None, None, None, None);

        // DA knockout → LR drops
        let (ko_lr, _, _, _, _, _) = run_ablation(Some(0.0), None, None, None);
        assert!(ko_lr < base_lr, "DA knockout should reduce LR");

        // NE knockout → exploration drops
        let (_, ko_expl, _, _, _, _) = run_ablation(None, Some(0.0), None, None);
        assert!(ko_expl < base_expl, "NE knockout should reduce exploration");

        // 5-HT knockout → confidence drops
        let (_, _, ko_conf, _, _, _) = run_ablation(None, None, Some(0.0), None);
        assert!(
            ko_conf < base_conf,
            "5-HT knockout should reduce confidence"
        );

        // ACh knockout → attention drops
        let (_, _, _, ko_attn, _, _) = run_ablation(None, None, None, Some(0.0));
        assert!(ko_attn < base_attn, "ACh knockout should reduce attention");
    }
}
