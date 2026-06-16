// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Allostatic Stress Benchmark: chronic stress load accumulation and recovery.
//!
//! Models McEwen's allostatic load framework: sustained cortisol elevations
//! progressively degrade neuromodulator baselines (DA, 5-HT), with recovery
//! requiring extended low-stress periods (sleep).
//!
//! Science: McEwen (1998) — Protective and damaging effects of stress mediators.
//!
//! 4 conditions:
//! - **Baseline**: 100 cycles at low cortisol (0.3) — load stays trivial
//! - **Acute stress**: 50 high cortisol + 50 rest — partial recovery
//! - **Chronic stress**: 200 cycles moderate-high cortisol — baseline depression
//! - **Burnout + recovery**: 300 high cortisol → burnout, then 200 sleep cycles → recovery

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;

/// Lightweight transmitter model for allostatic stress simulations.
#[derive(Clone)]
struct Transmitter {
    level: f32,
    baseline: f32,
    reuptake_rate: f32,
}

impl Transmitter {
    fn new() -> Self {
        Self {
            level: 0.5,
            baseline: 0.5,
            reuptake_rate: 0.1,
        }
    }

    fn produce(&mut self, amount: f32) {
        self.level = (self.level + amount).clamp(0.0, 1.0);
    }

    fn reuptake(&mut self) {
        self.level += (self.baseline - self.level) * self.reuptake_rate;
        self.level = self.level.clamp(0.0, 1.0);
    }
}

/// Lightweight bath model with DA, 5-HT, and Adenosine plus allostatic load tracking.
#[derive(Clone)]
struct Bath {
    da: Transmitter,
    sht: Transmitter,
    adenosine: Transmitter,
    allostatic_load: f32,
    allostatic_recovery_cycles: u32,
}

impl Bath {
    fn new() -> Self {
        Self {
            da: Transmitter::new(),
            sht: Transmitter::new(),
            adenosine: Transmitter::new(),
            allostatic_load: 0.0,
            allostatic_recovery_cycles: 0,
        }
    }

    /// Accumulate allostatic load from cortisol exposure.
    ///
    /// Mirrors the real symthaea-neuromodulators logic:
    /// - cortisol > 0.4 increases load
    /// - Natural decay each cycle
    /// - High load depresses DA/5-HT baselines
    /// - Recovery requires sleep + low load for 100 consecutive cycles
    fn accumulate_allostatic_load(&mut self, cortisol: f32, is_sleep: bool) {
        // Cortisol-driven load accumulation
        if cortisol > 0.4 {
            self.allostatic_load += (cortisol - 0.4) * 0.03;
        }

        // Natural decay (faster during sleep — restorative processes)
        if is_sleep {
            self.allostatic_load -= 0.005;
        } else {
            self.allostatic_load -= 0.002;
        }

        // Clamp to valid range
        self.allostatic_load = self.allostatic_load.clamp(0.0, 1.0);

        // Burnout regime: cap baselines hard
        if self.allostatic_load > 0.8 {
            if self.da.baseline > 0.35 {
                self.da.baseline = 0.35;
            }
            if self.sht.baseline > 0.35 {
                self.sht.baseline = 0.35;
            }
        }
        // Chronic stress regime: depress baselines proportionally
        else if self.allostatic_load > 0.5 {
            let depression = (self.allostatic_load - 0.5) * 0.02;
            self.da.baseline = (self.da.baseline - depression).max(0.1);
            self.sht.baseline = (self.sht.baseline - depression).max(0.1);
        }

        // Recovery: sleep + low load for 100 consecutive cycles → baselines recover
        if is_sleep && self.allostatic_load < 0.5 {
            self.allostatic_recovery_cycles += 1;
        } else {
            self.allostatic_recovery_cycles = 0;
        }

        if self.allostatic_recovery_cycles >= 100 {
            self.da.baseline = (self.da.baseline + 0.005).min(0.5);
            self.sht.baseline = (self.sht.baseline + 0.005).min(0.5);
        }
    }

    /// Run one cycle: produce neurotransmitters, reuptake, then accumulate load.
    fn step(&mut self, cortisol: f32, is_sleep: bool) {
        // Tonic production proportional to baseline
        self.da.produce(self.da.baseline * 0.05);
        self.sht.produce(self.sht.baseline * 0.05);
        // Adenosine accumulates during wake, clears during sleep
        if is_sleep {
            self.adenosine.produce(-0.02);
        } else {
            self.adenosine.produce(0.01);
        }

        self.da.reuptake();
        self.sht.reuptake();
        self.adenosine.reuptake();

        self.accumulate_allostatic_load(cortisol, is_sleep);
    }
}

/// Allostatic stress benchmark: chronic load accumulation and recovery dynamics.
///
/// Tests that sustained cortisol elevations degrade neuromodulator baselines
/// and that recovery requires extended restorative periods.
/// Science: McEwen (1998).
pub struct AllostaticStressBenchmark;

impl PsychBenchmark for AllostaticStressBenchmark {
    fn name(&self) -> &str {
        "Neuromod::AllostaticStress"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let mut result = BenchmarkResult::new(self.name(), None);
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        // ── Condition 1: Baseline ──
        // 100 cycles at cortisol=0.3 → allostatic load stays < 0.1
        {
            let mut bath = Bath::new();
            let mut peak_load: f32 = 0.0;
            for _ in 0..100 {
                bath.step(0.3, false);
                peak_load = peak_load.max(bath.allostatic_load);
            }
            result.insert(
                "baseline_allostatic_load_peak",
                MetricValue::from_samples(&[peak_load as f64]),
            );
            result.insert(
                "baseline_da_baseline_final",
                MetricValue::from_samples(&[bath.da.baseline as f64]),
            );
            result.insert(
                "baseline_sht_baseline_final",
                MetricValue::from_samples(&[bath.sht.baseline as f64]),
            );
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx,
                    condition: "baseline".to_string(),
                    correct: peak_load < 0.3,
                    rt_ticks: 0.0,
                    similarity: peak_load as f64,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                trial_idx += 1;
            }
        }

        // ── Condition 2: Acute stress ──
        // 50 cycles cortisol=0.7 → load rises; then 50 rest cycles cortisol=0.2 → partial recovery
        {
            let mut bath = Bath::new();
            let mut peak_load: f32 = 0.0;
            for _ in 0..50 {
                bath.step(0.7, false);
                peak_load = peak_load.max(bath.allostatic_load);
            }
            let load_after_stress = bath.allostatic_load;
            for _ in 0..50 {
                bath.step(0.2, false);
            }
            result.insert(
                "acute_allostatic_load_peak",
                MetricValue::from_samples(&[peak_load as f64]),
            );
            result.insert(
                "acute_load_after_stress",
                MetricValue::from_samples(&[load_after_stress as f64]),
            );
            result.insert(
                "acute_load_after_rest",
                MetricValue::from_samples(&[bath.allostatic_load as f64]),
            );
            result.insert(
                "acute_da_baseline_final",
                MetricValue::from_samples(&[bath.da.baseline as f64]),
            );
            result.insert(
                "acute_sht_baseline_final",
                MetricValue::from_samples(&[bath.sht.baseline as f64]),
            );
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx,
                    condition: "acute_stress".to_string(),
                    correct: true,
                    rt_ticks: 0.0,
                    similarity: bath.allostatic_load as f64,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                trial_idx += 1;
            }
        }

        // ── Condition 3: Chronic stress ──
        // 200 cycles cortisol=0.6 → load > 0.5, DA/5-HT baselines depressed
        {
            let mut bath = Bath::new();
            let mut peak_load: f32 = 0.0;
            for _ in 0..200 {
                bath.step(0.6, false);
                peak_load = peak_load.max(bath.allostatic_load);
            }
            result.insert(
                "chronic_allostatic_load_peak",
                MetricValue::from_samples(&[peak_load as f64]),
            );
            result.insert(
                "chronic_da_baseline_final",
                MetricValue::from_samples(&[bath.da.baseline as f64]),
            );
            result.insert(
                "chronic_sht_baseline_final",
                MetricValue::from_samples(&[bath.sht.baseline as f64]),
            );
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx,
                    condition: "chronic_stress".to_string(),
                    correct: bath.allostatic_load < 0.8,
                    rt_ticks: 0.0,
                    similarity: bath.allostatic_load as f64,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                trial_idx += 1;
            }
        }

        // ── Condition 4: Burnout + recovery ──
        // 300 cycles cortisol=0.7 → burnout (load > 0.8)
        // then 200 sleep cycles cortisol=0.2 → recovery trajectory
        {
            let mut bath = Bath::new();
            let mut peak_load: f32 = 0.0;
            for _ in 0..300 {
                bath.step(0.7, false);
                peak_load = peak_load.max(bath.allostatic_load);
            }
            let da_at_burnout = bath.da.baseline;
            let _sht_at_burnout = bath.sht.baseline;
            let mut recovery_cycles_needed: u32 = 0;
            let burnout_load = bath.allostatic_load;
            for i in 0..200 {
                bath.step(0.2, true);
                if recovery_cycles_needed == 0 && bath.da.baseline > da_at_burnout {
                    recovery_cycles_needed = i + 1;
                }
            }
            result.insert(
                "burnout_allostatic_load_peak",
                MetricValue::from_samples(&[peak_load as f64]),
            );
            result.insert(
                "burnout_load_at_peak",
                MetricValue::from_samples(&[burnout_load as f64]),
            );
            result.insert(
                "burnout_da_baseline_final",
                MetricValue::from_samples(&[bath.da.baseline as f64]),
            );
            result.insert(
                "burnout_sht_baseline_final",
                MetricValue::from_samples(&[bath.sht.baseline as f64]),
            );
            result.insert(
                "burnout_recovery_cycles_needed",
                MetricValue::from_samples(&[recovery_cycles_needed as f64]),
            );
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx,
                    condition: "burnout_recovery".to_string(),
                    correct: bath.allostatic_load < burnout_load,
                    rt_ticks: 0.0,
                    similarity: bath.allostatic_load as f64,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                trial_idx += 1;
            }
        }

        if config.trial_trace {
            result.trial_trace = trace;
        }
        let _ = trial_idx;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            citation: "McEwen (1998) — Protective and damaging effects of stress mediators",
            paradigm: "Allostatic load / chronic stress",
            year: 1998,
            doi: Some("10.1056/NEJM199801153380307"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::config::BenchmarkConfig;

    #[test]
    fn test_benchmark_runs() {
        let config = BenchmarkConfig::default();
        let bench = AllostaticStressBenchmark;
        let result = bench.run(&config);
        // All metrics should be finite
        for (key, val) in &result.metrics {
            assert!(
                val.mean.is_finite(),
                "Non-finite metric: {key} = {}",
                val.mean
            );
        }
        // Should have metrics from all 4 conditions
        assert!(
            result.metrics.len() >= 14,
            "Expected at least 14 metrics, got {}",
            result.metrics.len()
        );
    }

    #[test]
    fn test_acute_recovery() {
        // After acute stress (50 cycles cortisol=0.7) + rest (50 cycles cortisol=0.2),
        // allostatic load should drop compared to peak.
        let mut bath = Bath::new();
        for _ in 0..50 {
            bath.step(0.7, false);
        }
        let peak_load = bath.allostatic_load;
        assert!(
            peak_load > 0.0,
            "Load should rise under acute stress: got {peak_load}"
        );

        for _ in 0..50 {
            bath.step(0.2, false);
        }
        assert!(
            bath.allostatic_load < peak_load,
            "Load should drop after rest: peak={peak_load}, after_rest={}",
            bath.allostatic_load
        );
    }

    #[test]
    fn test_chronic_depression() {
        // 200 cycles at cortisol=0.6 should depress DA and 5-HT baselines below 0.5.
        let mut bath = Bath::new();
        for _ in 0..200 {
            bath.step(0.6, false);
        }
        assert!(
            bath.da.baseline < 0.5,
            "DA baseline should be depressed: got {}",
            bath.da.baseline
        );
        assert!(
            bath.sht.baseline < 0.5,
            "5-HT baseline should be depressed: got {}",
            bath.sht.baseline
        );
        assert!(
            bath.allostatic_load > 0.5,
            "Allostatic load should exceed 0.5 under chronic stress: got {}",
            bath.allostatic_load
        );
    }

    #[test]
    fn test_burnout_threshold() {
        // 300 cycles at cortisol=0.7 should push allostatic load above 0.8.
        let mut bath = Bath::new();
        let mut peak_load: f32 = 0.0;
        for _ in 0..300 {
            bath.step(0.7, false);
            peak_load = peak_load.max(bath.allostatic_load);
        }
        assert!(
            peak_load > 0.8,
            "Peak allostatic load should exceed 0.8 under sustained high cortisol: got {peak_load}"
        );
        // DA and 5-HT baselines should be capped at 0.35 in burnout
        assert!(
            bath.da.baseline <= 0.35 + f32::EPSILON,
            "DA baseline should be capped at 0.35 in burnout: got {}",
            bath.da.baseline
        );
        assert!(
            bath.sht.baseline <= 0.35 + f32::EPSILON,
            "5-HT baseline should be capped at 0.35 in burnout: got {}",
            bath.sht.baseline
        );
    }

    #[test]
    fn test_normative_ranges() {
        // Check that metrics from all conditions are in reasonable ranges.
        let config = BenchmarkConfig::default();
        let bench = AllostaticStressBenchmark;
        let result = bench.run(&config);

        // Baseline condition: load should stay very low
        let baseline_peak = result.metrics["baseline_allostatic_load_peak"].mean;
        assert!(
            baseline_peak < 0.1,
            "Baseline allostatic load peak should be < 0.1: got {baseline_peak}"
        );

        // Baseline DA/5-HT should remain at 0.5 (unperturbed)
        let baseline_da = result.metrics["baseline_da_baseline_final"].mean;
        assert!(
            (baseline_da - 0.5).abs() < 0.01,
            "Baseline DA should remain ~0.5: got {baseline_da}"
        );

        // Burnout peak load should be high
        let burnout_peak = result.metrics["burnout_allostatic_load_peak"].mean;
        assert!(
            burnout_peak > 0.8,
            "Burnout peak should exceed 0.8: got {burnout_peak}"
        );

        // All metric values should be in [0, 1] range (or recovery_cycles in [0, 200])
        for (key, val) in &result.metrics {
            if key.contains("recovery_cycles") {
                assert!(
                    val.mean >= 0.0 && val.mean <= 200.0,
                    "Recovery cycles out of range: {key} = {}",
                    val.mean
                );
            } else {
                assert!(
                    val.mean >= 0.0 && val.mean <= 1.0,
                    "Metric out of [0, 1] range: {key} = {}",
                    val.mean
                );
            }
        }
    }
}
