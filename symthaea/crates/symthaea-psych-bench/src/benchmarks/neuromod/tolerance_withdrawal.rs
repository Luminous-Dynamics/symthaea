// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tolerance/Withdrawal Validation Benchmark
//!
//! Verifies per-transmitter tolerance curves develop under sustained injection
//! and withdrawal rebound occurs after cessation.
//!
//! Science: Koob & Le Moal (2001) — Allostatic model of addiction
//!          Gainetdinov et al. (2004) — GPCR desensitization kinetics

use crate::harness::PsychBenchmark;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use std::collections::BTreeMap;
use symthaea_neuromodulators::{NeuromodulatorBath, NeuromodulatorInputs};

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

/// Per-transmitter protocol parameters.
struct ToleranceProtocol {
    name: &'static str,
    target: &'static str,
    /// Cycles of sustained injection before measuring tolerance
    exposure_cycles: usize,
    /// Cycles after cessation to observe withdrawal
    withdrawal_observation: usize,
}

const PROTOCOLS: &[ToleranceProtocol] = &[
    ToleranceProtocol {
        name: "da",
        target: "da",
        exposure_cycles: 75, // onset=15, so 75 cycles is well past tolerance onset
        withdrawal_observation: 60,
    },
    ToleranceProtocol {
        name: "ne",
        target: "ne",
        exposure_cycles: 85, // onset=25
        withdrawal_observation: 40,
    },
    ToleranceProtocol {
        name: "sht",
        target: "sht",
        exposure_cycles: 90, // onset=30
        withdrawal_observation: 70,
    },
    ToleranceProtocol {
        name: "gaba",
        target: "gaba",
        exposure_cycles: 60, // onset=10
        withdrawal_observation: 45,
    },
];

/// Access the right transmitter from the bath by target name.
fn get_transmitter(bath: &NeuromodulatorBath, target: &str) -> (bool, bool, f32) {
    match target {
        "da" => (
            bath.dopamine.is_tolerant(),
            bath.dopamine.is_in_withdrawal(),
            bath.dopamine.receptor_sensitivity,
        ),
        "ne" => (
            bath.noradrenaline.is_tolerant(),
            bath.noradrenaline.is_in_withdrawal(),
            bath.noradrenaline.receptor_sensitivity,
        ),
        "sht" => (
            bath.serotonin.is_tolerant(),
            bath.serotonin.is_in_withdrawal(),
            bath.serotonin.receptor_sensitivity,
        ),
        "gaba" => (
            bath.gaba.is_tolerant(),
            bath.gaba.is_in_withdrawal(),
            bath.gaba.receptor_sensitivity,
        ),
        _ => (false, false, 1.0),
    }
}

/// Run the tolerance/withdrawal protocol for one transmitter.
fn run_protocol(proto: &ToleranceProtocol) -> (bool, f32, bool, f32) {
    let inputs = neutral_inputs();
    let mut bath = NeuromodulatorBath::default();

    // Phase A: Warmup (30 cycles)
    for _ in 0..30 {
        bath.update(&inputs);
    }
    let (_, _, early_sensitivity) = get_transmitter(&bath, proto.target);

    // Phase B: Sustained injection
    for _ in 0..proto.exposure_cycles {
        // Re-inject each cycle to maintain high level (half_life=500 = long duration)
        if bath.active_injections.len() < 4 {
            bath.inject(proto.target, 0.5, 500);
        }
        bath.update(&inputs);
    }
    let (tolerance_detected, _, late_sensitivity) = get_transmitter(&bath, proto.target);
    let sensitivity_decrease = late_sensitivity / early_sensitivity.max(0.001);

    // Phase C: Withdrawal (clear injections + deplete to trigger rebound)
    // The bath's withdrawal mechanism requires level < baseline while still tolerant.
    // We simulate abrupt cessation by clearing injections and pushing level below baseline,
    // mimicking the sudden drop that triggers withdrawal in biological systems.
    bath.clear_injections();
    // Deplete the target transmitter to trigger withdrawal detection
    match proto.target {
        "da" => bath.clamp_all_levels(Some(0.2), None, None, None, None, None, None),
        "ne" => bath.clamp_all_levels(None, Some(0.2), None, None, None, None, None),
        "sht" => bath.clamp_all_levels(None, None, Some(0.2), None, None, None, None),
        "gaba" => bath.clamp_all_levels(None, None, None, None, Some(0.1), None, None),
        _ => {}
    }
    // One reuptake cycle with level below baseline to trigger withdrawal_cycles
    bath.update(&inputs);
    let mut withdrawal_detected = false;
    let mut peak_sensitivity = late_sensitivity;
    for _ in 0..proto.withdrawal_observation {
        bath.update(&inputs);
        let (_, in_withdrawal, sens) = get_transmitter(&bath, proto.target);
        if in_withdrawal {
            withdrawal_detected = true;
        }
        if sens > peak_sensitivity {
            peak_sensitivity = sens;
        }
    }

    (
        tolerance_detected,
        sensitivity_decrease,
        withdrawal_detected,
        peak_sensitivity,
    )
}

/// Tolerance/Withdrawal Validation Benchmark.
///
/// Validates that sustained pharmacological injection produces receptor
/// desensitization (tolerance) and that cessation triggers withdrawal
/// rebound (sensitization).
pub struct ToleranceWithdrawalBenchmark;

impl PsychBenchmark for ToleranceWithdrawalBenchmark {
    fn name(&self) -> &str {
        "ToleranceWithdrawal"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), None);
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        let mut tolerance_count = 0u32;
        let mut withdrawal_count = 0u32;

        for proto in PROTOCOLS {
            let (tolerance, sensitivity_ratio, withdrawal, peak_sens) = run_protocol(proto);

            result.insert(
                format!("{}_tolerance_detected", proto.name),
                MetricValue::from_samples(&[if tolerance { 1.0 } else { 0.0 }]),
            );
            result.insert(
                format!("{}_sensitivity_ratio", proto.name),
                MetricValue::from_samples(&[sensitivity_ratio as f64]),
            );
            result.insert(
                format!("{}_withdrawal_detected", proto.name),
                MetricValue::from_samples(&[if withdrawal { 1.0 } else { 0.0 }]),
            );
            result.insert(
                format!("{}_withdrawal_peak_sensitivity", proto.name),
                MetricValue::from_samples(&[peak_sens as f64]),
            );

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx,
                    condition: proto.name.to_string(),
                    correct: tolerance,
                    rt_ticks: 0.0,
                    similarity: sensitivity_ratio as f64,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                trial_idx += 1;
            }

            if tolerance {
                tolerance_count += 1;
            }
            if withdrawal {
                withdrawal_count += 1;
            }
        }

        result.insert(
            "tolerance_count",
            MetricValue::from_samples(&[tolerance_count as f64]),
        );
        result.insert(
            "withdrawal_count",
            MetricValue::from_samples(&[withdrawal_count as f64]),
        );

        if config.trial_trace {
            result.trial_trace = trace;
        }
        let _ = trial_idx;

        result.conditions = PROTOCOLS.len();
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
    fn test_tolerance_withdrawal_runs() {
        let bench = ToleranceWithdrawalBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "ToleranceWithdrawal");
        assert!(result.metrics.len() >= 10);
    }

    #[test]
    fn test_da_tolerance_develops() {
        let bench = ToleranceWithdrawalBenchmark;
        let result = bench.run(&config());
        let detected = result.metrics.get("da_tolerance_detected").unwrap().mean;
        assert!(
            (detected - 1.0).abs() < f64::EPSILON,
            "DA tolerance should develop under sustained injection"
        );
    }

    #[test]
    fn test_ne_tolerance_develops() {
        let bench = ToleranceWithdrawalBenchmark;
        let result = bench.run(&config());
        let detected = result.metrics.get("ne_tolerance_detected").unwrap().mean;
        assert!(
            (detected - 1.0).abs() < f64::EPSILON,
            "NE tolerance should develop under sustained injection"
        );
    }

    #[test]
    fn test_sht_tolerance_develops() {
        let bench = ToleranceWithdrawalBenchmark;
        let result = bench.run(&config());
        let detected = result.metrics.get("sht_tolerance_detected").unwrap().mean;
        assert!(
            (detected - 1.0).abs() < f64::EPSILON,
            "5-HT tolerance should develop under sustained injection"
        );
    }

    #[test]
    fn test_gaba_tolerance_develops() {
        let bench = ToleranceWithdrawalBenchmark;
        let result = bench.run(&config());
        let detected = result.metrics.get("gaba_tolerance_detected").unwrap().mean;
        assert!(
            (detected - 1.0).abs() < f64::EPSILON,
            "GABA tolerance should develop under sustained injection"
        );
    }

    #[test]
    fn test_withdrawal_rebound_detected() {
        let bench = ToleranceWithdrawalBenchmark;
        let result = bench.run(&config());
        let count = result.metrics.get("withdrawal_count").unwrap().mean;
        assert!(
            count >= 3.0,
            "At least 3 of 4 transmitters should show withdrawal: got {count}"
        );
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = ToleranceWithdrawalBenchmark;
        let result = bench.run(&config());
        for (name, metric) in &result.metrics {
            assert!(
                metric.mean.is_finite(),
                "Metric '{name}' should be finite, got {}",
                metric.mean
            );
        }
    }
}
