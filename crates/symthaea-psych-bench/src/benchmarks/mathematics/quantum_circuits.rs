// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quantum Circuit Assessment.
//!
//! Tests the statevector quantum circuit simulator against circuits with
//! analytically known output distributions, using `QuantumCircuit` from
//! symthaea-core.
//!
//! ## Circuits
//!
//! 1. **Superposition**: H|0⟩ → |+⟩ = (|0⟩+|1⟩)/√2.
//!    Verify p(0) = p(1) = 0.5 ± 1e-9.
//!
//! 2. **Bell State**: (H⊗I)·CNOT|00⟩ → (|00⟩+|11⟩)/√2.
//!    Verify p(|00⟩) = p(|11⟩) = 0.5, p(|01⟩) = p(|10⟩) = 0.0.
//!
//! 3. **Phase Flip**: Z|+⟩ → |−⟩ = (|0⟩−|1⟩)/√2.
//!    Verify amplitudes flip sign on |1⟩; probabilities unchanged (still 0.5/0.5).
//!
//! 4. **Normalization Invariant**: All circuits must yield total probability = 1.0.
//!
//! Human baselines (Nielsen & Chuang, 2010):
//! - superposition_accuracy:    1.00 (exact — deterministic statevector simulation)
//! - bell_state_accuracy:       1.00 (exact)
//! - phase_flip_accuracy:       1.00 (exact)
//! - normalization_preserved:   1.00 (exact — invariant of unitary gates)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::quantum_circuit::QuantumCircuit;

/// Quantum Circuit Simulation Assessment benchmark.
pub struct QuantumCircuitsBenchmark;

struct TrialResult {
    superposition_correct: f64,
    bell_state_correct: f64,
    phase_flip_correct: f64,
    normalization_preserved: f64,
}

impl QuantumCircuitsBenchmark {
    fn run_trial(&self, _config: &BenchmarkConfig, _trial_idx: usize) -> TrialResult {
        // ── 1. Superposition: H|0⟩ → |+⟩ ────────────────────────────────
        let state_plus = QuantumCircuit::new(1).h(0).execute();
        let p0 = state_plus.probability(0); // |0⟩
        let p1 = state_plus.probability(1); // |1⟩
        let superposition_correct =
            if (p0 - 0.5).abs() < 1e-9 && (p1 - 0.5).abs() < 1e-9 { 1.0 } else { 0.0 };

        // ── 2. Bell State: (H⊗I)·CNOT|00⟩ → (|00⟩+|11⟩)/√2 ─────────────
        // Qubit ordering: qubit 0 = MSB. CNOT with control=0, target=1.
        // Basis: |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3.
        let bell = QuantumCircuit::new(2).h(0).cnot(0, 1).execute();
        let p00 = bell.probability(0b00); // |00⟩
        let p01 = bell.probability(0b01); // |01⟩
        let p10 = bell.probability(0b10); // |10⟩
        let p11 = bell.probability(0b11); // |11⟩
        let bell_state_correct = if (p00 - 0.5).abs() < 1e-9
            && p01 < 1e-9
            && p10 < 1e-9
            && (p11 - 0.5).abs() < 1e-9
        {
            1.0
        } else {
            0.0
        };

        // ── 3. Phase Flip: Z|+⟩ → |−⟩ ────────────────────────────────────
        // Start: H|0⟩ = |+⟩ = [1/√2, 1/√2]
        // After Z: [1/√2, -1/√2] = |−⟩
        // Probabilities unchanged: p(0)=0.5, p(1)=0.5 — but amplitude of |1⟩ is negated.
        let state_minus = QuantumCircuit::new(1).h(0).z(0).execute();
        let pm0 = state_minus.probability(0);
        let pm1 = state_minus.probability(1);
        // Verify probabilities are still 0.5/0.5 (phase flip doesn't change probabilities).
        let probs_unchanged = (pm0 - 0.5).abs() < 1e-9 && (pm1 - 0.5).abs() < 1e-9;
        // Verify the amplitude of |1⟩ is negative (real part < 0).
        let amp1_real = state_minus.amplitudes[1].0;
        let amp1_neg = amp1_real < -0.6; // should be -1/√2 ≈ -0.707
        let phase_flip_correct = if probs_unchanged && amp1_neg { 1.0 } else { 0.0 };

        // ── 4. Normalization preserved across multiple circuits ────────────
        let circuits_normalized = [
            QuantumCircuit::new(1).h(0).execute().total_probability(),
            QuantumCircuit::new(2).h(0).cnot(0, 1).execute().total_probability(),
            QuantumCircuit::new(1).h(0).z(0).execute().total_probability(),
            QuantumCircuit::new(2).h(0).h(1).execute().total_probability(),
            QuantumCircuit::new(1).x(0).h(0).execute().total_probability(),
        ];
        let normalization_preserved = if circuits_normalized
            .iter()
            .all(|&p| (p - 1.0).abs() < 1e-9)
        {
            1.0
        } else {
            0.0
        };

        TrialResult {
            superposition_correct,
            bell_state_correct,
            phase_flip_correct,
            normalization_preserved,
        }
    }
}

impl PsychBenchmark for QuantumCircuitsBenchmark {
    fn name(&self) -> &str {
        "Mathematics::QuantumCircuits"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Statevector Quantum Circuit Simulation Assessment",
            citation: "Nielsen & Chuang (2010)",
            year: 2010,
            doi: Some("10.1017/CBO9780511976667"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut super_scores = Vec::with_capacity(config.trials_per_condition);
        let mut bell_scores = Vec::with_capacity(config.trials_per_condition);
        let mut phase_scores = Vec::with_capacity(config.trials_per_condition);
        let mut norm_scores = Vec::with_capacity(config.trials_per_condition);

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            super_scores.push(r.superposition_correct);
            bell_scores.push(r.bell_state_correct);
            phase_scores.push(r.phase_flip_correct);
            norm_scores.push(r.normalization_preserved);
        }

        result.insert(
            "superposition_accuracy",
            MetricValue::from_samples(&super_scores),
        );
        result.insert("bell_state_accuracy", MetricValue::from_samples(&bell_scores));
        result.insert("phase_flip_accuracy", MetricValue::from_samples(&phase_scores));
        result.insert(
            "normalization_preserved",
            MetricValue::from_samples(&norm_scores),
        );

        result.conditions = 4;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_quantum_runs_and_has_metrics() {
        let result = QuantumCircuitsBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("superposition_accuracy"));
        assert!(result.metrics.contains_key("bell_state_accuracy"));
        assert!(result.metrics.contains_key("phase_flip_accuracy"));
        assert!(result.metrics.contains_key("normalization_preserved"));
    }

    #[test]
    fn test_quantum_metrics_finite() {
        let result = QuantumCircuitsBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_hadamard_superposition() {
        let state = QuantumCircuit::new(1).h(0).execute();
        let p0 = state.probability(0);
        let p1 = state.probability(1);
        assert!(
            (p0 - 0.5).abs() < 1e-9,
            "H|0⟩: p(|0⟩) = {:.10}, expected 0.5",
            p0
        );
        assert!(
            (p1 - 0.5).abs() < 1e-9,
            "H|0⟩: p(|1⟩) = {:.10}, expected 0.5",
            p1
        );
    }

    #[test]
    fn test_bell_state_entanglement() {
        let state = QuantumCircuit::new(2).h(0).cnot(0, 1).execute();
        let p00 = state.probability(0);
        let p11 = state.probability(3);
        assert!(
            (p00 - 0.5).abs() < 1e-9,
            "Bell state: p(|00⟩) = {:.10}, expected 0.5",
            p00
        );
        assert!(
            (p11 - 0.5).abs() < 1e-9,
            "Bell state: p(|11⟩) = {:.10}, expected 0.5",
            p11
        );
        assert!(
            state.probability(1) < 1e-9,
            "Bell state: p(|01⟩) should be 0"
        );
        assert!(
            state.probability(2) < 1e-9,
            "Bell state: p(|10⟩) should be 0"
        );
    }

    #[test]
    fn test_normalization_invariant() {
        let circuits = [
            QuantumCircuit::new(1).h(0).execute(),
            QuantumCircuit::new(2).h(0).cnot(0, 1).execute(),
            QuantumCircuit::new(1).x(0).execute(),
            QuantumCircuit::new(3).h(0).h(1).h(2).execute(),
        ];
        for (i, state) in circuits.iter().enumerate() {
            let total = state.total_probability();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "Circuit {}: total probability = {:.12}, expected 1.0",
                i,
                total
            );
        }
    }

    #[test]
    fn test_z_gate_flips_phase() {
        let state = QuantumCircuit::new(1).h(0).z(0).execute();
        // |−⟩ = (|0⟩ - |1⟩)/√2: amplitude[0] > 0, amplitude[1] < 0
        assert!(
            state.amplitudes[0].0 > 0.6,
            "Z|+⟩: amp[0].re = {:.4}, should be ~+0.707",
            state.amplitudes[0].0
        );
        assert!(
            state.amplitudes[1].0 < -0.6,
            "Z|+⟩: amp[1].re = {:.4}, should be ~-0.707",
            state.amplitudes[1].0
        );
    }
}
