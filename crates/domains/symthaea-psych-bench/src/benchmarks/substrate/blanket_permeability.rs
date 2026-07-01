// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Markov Blanket Permeability benchmark.
//!
//! Measures how dynamic blanket permeability affects consciousness metrics
//! under simulated threat vs safety conditions. Tests Friston's (2013)
//! prediction that Markov blanket thickness modulates information integration.
//!
//! ## Paradigm
//!
//! Simulates 100-cycle cognitive episodes under two conditions:
//! - **Safety**: High serotonin, oxytocin, flow; low threat/NE/ACh
//! - **Threat**: Low serotonin/oxytocin; high threat, NE, ACh
//!
//! Measures:
//! - `permeability_range`: Dynamic range between safe and threat (higher = more adaptive)
//! - `learning_rate_ratio`: Safe LR / Threat LR (should be > 1.0)
//! - `coalescence_latency`: Cycles to reach coalescence-ready under safety
//! - `recovery_time`: Cycles to reopen blanket after threat removal
//!
//! ## Theoretical Baselines
//!
//! - permeability_range: 0.40 (SD≈0.10) — Friston (2013) adaptive blanket
//! - learning_rate_ratio: 1.50 (SD≈0.20) — Yu & Dayan (2005) uncertainty modulation
//! - coalescence_latency: 40 cycles (SD≈10) — Kirchhoff et al. (2018)
//! - recovery_time: 30 cycles (SD≈10) — allostatic recovery (Sterling & Eyer, 1988)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_fep::{ActiveInferenceAgentConfig, EnhancedFEPBridge, PermeabilityInputs};

/// Blanket Permeability benchmark.
pub struct BlanketPermeabilityBenchmark;

struct TrialResult {
    safe_permeability: f64,
    threat_permeability: f64,
    permeability_range: f64,
    safe_learning_rate: f64,
    threat_learning_rate: f64,
    learning_rate_ratio: f64,
    coalescence_latency: usize,
    recovery_time: usize,
}

impl BlanketPermeabilityBenchmark {
    fn safe_inputs() -> PermeabilityInputs {
        PermeabilityInputs {
            serotonin: 0.85,
            oxytocin: 0.75,
            flow_state: 0.70,
            threat_level: 0.0,
            noradrenaline: 0.15,
            acetylcholine: 0.20,
            peer_trust: 0.80,
        }
    }

    fn threat_inputs() -> PermeabilityInputs {
        PermeabilityInputs {
            serotonin: 0.10,
            oxytocin: 0.10,
            flow_state: 0.0,
            threat_level: 0.85,
            noradrenaline: 0.80,
            acetylcholine: 0.70,
            peer_trust: 0.15,
        }
    }

    fn run_trial(config: &BenchmarkConfig) -> TrialResult {
        let agent_config = ActiveInferenceAgentConfig::default();
        let mut bridge = EnhancedFEPBridge::new(agent_config, 4);
        let safe = Self::safe_inputs();
        let threat = Self::threat_inputs();

        let lapse = config.lapse_rate;

        // ── Phase 1: Safety (50 cycles) ─────────────────────────────────
        for _ in 0..50 {
            bridge.update_blanket_permeability(&safe);
            bridge.cycle(0.6, 0.5, 0.7, 0.4);
        }
        let safe_permeability = bridge.blanket.permeability().effective;
        let safe_learning_rate = bridge.blanket.modulate_learning_rate(1.0);

        // Measure coalescence latency (cycles to first coalescence-ready)
        let mut coalescence_latency = 50; // Already ran 50 cycles
        if !bridge.blanket.coalescence_ready(0.6) {
            // Run more cycles until ready or max 150
            for i in 0..100 {
                bridge.update_blanket_permeability(&safe);
                bridge.cycle(0.6, 0.5, 0.7, 0.4);
                if bridge.blanket.coalescence_ready(0.6) {
                    coalescence_latency = 50 + i + 1;
                    break;
                }
            }
        }

        // ── Phase 2: Threat (50 cycles) ─────────────────────────────────
        for _ in 0..50 {
            bridge.update_blanket_permeability(&threat);
            bridge.cycle(0.3, 0.2, 0.4, 0.3);
        }
        let threat_permeability = bridge.blanket.permeability().effective;
        let threat_learning_rate = bridge.blanket.modulate_learning_rate(1.0);

        // ── Phase 3: Recovery (threat → safety, measure reopen time) ────
        let mut recovery_time = 100; // Max
        for i in 0..100 {
            bridge.update_blanket_permeability(&safe);
            bridge.cycle(0.5, 0.5, 0.5, 0.5);
            if bridge.blanket.permeability().effective > 0.5 {
                recovery_time = i + 1;
                break;
            }
        }

        // Apply lapse rate noise
        let noise = |base: f64| base * (1.0 - lapse * 0.3);

        let permeability_range = (safe_permeability - threat_permeability).abs();
        let learning_rate_ratio = if threat_learning_rate > 0.01 {
            safe_learning_rate / threat_learning_rate
        } else {
            1.0
        };

        TrialResult {
            safe_permeability: noise(safe_permeability),
            threat_permeability: noise(threat_permeability),
            permeability_range: noise(permeability_range),
            safe_learning_rate: noise(safe_learning_rate),
            threat_learning_rate: noise(threat_learning_rate),
            learning_rate_ratio: noise(learning_rate_ratio),
            coalescence_latency,
            recovery_time,
        }
    }
}

impl PsychBenchmark for BlanketPermeabilityBenchmark {
    fn name(&self) -> &str {
        "BlanketPermeability"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let n_trials = config.trials_per_condition.max(5);
        let mut trial_results: Vec<TrialResult> = Vec::with_capacity(n_trials);

        for _ in 0..n_trials {
            trial_results.push(Self::run_trial(config));
        }

        let collect =
            |f: fn(&TrialResult) -> f64| -> Vec<f64> { trial_results.iter().map(f).collect() };

        let mut result = BenchmarkResult::new(self.name(), None);
        result.conditions = 1;
        result.trials_per_condition = n_trials;
        result.metrics.insert(
            "permeability_range".into(),
            MetricValue::from_samples(&collect(|r| r.permeability_range)),
        );
        result.metrics.insert(
            "learning_rate_ratio".into(),
            MetricValue::from_samples(&collect(|r| r.learning_rate_ratio)),
        );
        result.metrics.insert(
            "coalescence_latency".into(),
            MetricValue::from_samples(&collect(|r| r.coalescence_latency as f64)),
        );
        result.metrics.insert(
            "recovery_time".into(),
            MetricValue::from_samples(&collect(|r| r.recovery_time as f64)),
        );
        result.metrics.insert(
            "safe_permeability".into(),
            MetricValue::from_samples(&collect(|r| r.safe_permeability)),
        );
        result.metrics.insert(
            "threat_permeability".into(),
            MetricValue::from_samples(&collect(|r| r.threat_permeability)),
        );
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Markov Blanket Permeability — neuromod-coupled dynamic boundary",
            citation: "Friston (2013) Life as we know it; Kirchhoff et al. (2018) Markov blankets of life",
            year: 2013,
            doi: Some("10.1098/rsif.2013.0475"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blanket_permeability_benchmark_runs() {
        let benchmark = BlanketPermeabilityBenchmark;
        let config = BenchmarkConfig::default();
        let result = benchmark.run(&config);

        assert_eq!(result.benchmark_name, "BlanketPermeability");
        assert_eq!(result.domain, "Substrate");

        // Permeability range should be > 0 (blanket is adaptive)
        let range = result.metrics.get("permeability_range").unwrap();
        assert!(
            range.mean > 0.1,
            "Permeability range should be substantial: {}",
            range.mean
        );

        // Learning rate ratio should be > 1.0 (safe > threat)
        let lr_ratio = result.metrics.get("learning_rate_ratio").unwrap();
        assert!(
            lr_ratio.mean > 1.0,
            "Safe LR should exceed threat LR: {}",
            lr_ratio.mean
        );

        // Recovery time should be finite and reasonable
        let recovery = result.metrics.get("recovery_time").unwrap();
        assert!(
            recovery.mean < 80.0,
            "Recovery should happen within 80 cycles: {}",
            recovery.mean
        );
    }
}