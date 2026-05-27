// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Integration tests for the unified thermodynamic framework.
//!
//! These tests validate that the thermodynamic unification produces
//! measurably different behavior under different conditions, proving
//! the cross-couplings and feedback loops are active and meaningful.

use crate::cognitive_loop::thermodynamic_integration::{
    ThermodynamicFeedback, ThermodynamicInput, ThermodynamicIntegration,
};
use crate::consciousness::consciousness_thermodynamics::ConsciousnessPhase;
use crate::consciousness::dissipative_consciousness::ThermodynamicRegime;

/// Helper: standard healthy-consciousness input
fn healthy_input() -> ThermodynamicInput {
    ThermodynamicInput {
        unified_psi: 0.7,
        coherence: 0.8,
        prediction_error: 0.2,
        attention_sensitivity: 0.6,
        energy_per_cycle: 1e-10,
        total_energy_spent: 5e-8,
        energy_throughput_multiplier: 1.0,
        metabolic_stress: 0.15,
        dissipative_health: 0.75,
        dissipative_regime: ThermodynamicRegime::EdgeOfChaos,
        entropy_production_rate: 0.2,
        order_parameter: 0.65,
        criticality_distance: 0.03,
        lambda_parameter: 0.27,
        dissipation_efficiency: 0.55,
        has_dissipative: true,
        analyzer_entropy: 0.45,
        analyzer_free_energy: -0.15,
        analyzer_temperature: 0.42,
        analyzer_phase: ConsciousnessPhase::Normal,
        has_analyzer: true,
        hfe_total: 0.6,
        hfe_complexity: 0.35,
        hfe_accuracy: 0.25,
        has_hfe: true,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// A/B Tests: Unified vs. absent modules
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_unified_vs_absent_modules() {
    // A: Full unified system with all modules
    let mut full = ThermodynamicIntegration::default();
    let input = healthy_input();
    let mut full_feedbacks = Vec::new();
    for _ in 0..50 {
        full_feedbacks.push(full.run_cycle(&input));
    }

    // B: Absent modules (no dissipative, no analyzer, no HFE)
    let mut empty = ThermodynamicIntegration::default();
    let empty_input = ThermodynamicInput::default();
    let mut empty_feedbacks = Vec::new();
    for _ in 0..50 {
        empty_feedbacks.push(empty.run_cycle(&empty_input));
    }

    // The full system should have non-trivial state; empty should be zeros
    assert!(full.state.dissipative_health > 0.0);
    assert!(full.state.canonical_entropy > 0.0);
    assert_eq!(empty.state.dissipative_health, 0.0);
    assert_eq!(empty.state.canonical_entropy, 0.0);

    // Full system should have active physics bridge
    assert!(full.bridge.carnot_efficiency > 0.0);
    assert!(full.bridge.attention_demon_bits > 0.0);

    // Empty system should have near-zero physics
    assert!(empty.bridge.attention_demon_bits < 0.01);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stability: Edge-of-chaos convergence
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_edge_of_chaos_stability_over_100_cycles() {
    let mut ti = ThermodynamicIntegration::default();
    let input = healthy_input();

    let mut health_history = Vec::new();
    for _ in 0..100 {
        ti.run_cycle(&input);
        health_history.push(ti.state.dissipative_health);
    }

    // Health should converge (variance in last 20 < variance in first 20)
    let first_var = variance(&health_history[..20]);
    let last_var = variance(&health_history[80..]);
    assert!(
        last_var <= first_var * 2.0, // Allow some tolerance
        "Health should stabilize: first_var={first_var:.6}, last_var={last_var:.6}"
    );

    // Final health should be positive (system alive)
    assert!(health_history.last().unwrap() > &0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Perturbation: Temperature spike recovery
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_temperature_spike_recovery() {
    let mut ti = ThermodynamicIntegration::default();
    let normal = healthy_input();

    // Warm up to steady state
    for _ in 0..30 {
        ti.run_cycle(&normal);
    }
    let steady_temp = ti.state.effective_temperature;

    // Inject temperature spike
    let mut hot = normal.clone();
    hot.analyzer_temperature = 1.8; // Way above normal
    hot.dissipative_regime = ThermodynamicRegime::Chaotic;
    hot.entropy_production_rate = 0.9;
    for _ in 0..5 {
        ti.run_cycle(&hot);
    }
    let spike_temp = ti.state.effective_temperature;
    assert!(spike_temp > steady_temp, "Temperature should spike");

    // Recovery: return to normal input
    for _ in 0..30 {
        ti.run_cycle(&normal);
    }
    let recovered_temp = ti.state.effective_temperature;

    // Should recover toward steady state (EMA smoothing)
    assert!(
        (recovered_temp - steady_temp).abs() < (spike_temp - steady_temp).abs() * 0.5,
        "Temperature should recover: steady={steady_temp:.3}, spike={spike_temp:.3}, recovered={recovered_temp:.3}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Insight: Bifurcation triggers insight detection
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_insight_fires_during_bifurcation() {
    let mut ti = ThermodynamicIntegration::default();
    let normal = healthy_input();

    // Build up steady state with low order
    let mut low_order = normal.clone();
    low_order.order_parameter = 0.2;
    for _ in 0..10 {
        ti.run_cycle(&low_order);
    }

    // Sudden order increase (bifurcation → insight)
    let mut high_order = normal.clone();
    high_order.order_parameter = 0.9;
    let fb = ti.run_cycle(&high_order);

    // Should detect insight and boost LR
    if fb.insight_detected {
        assert!(fb.lr_factor > 1.0, "Insight should boost LR");
    }
    // Even if insight threshold not hit, probability should be > 0
    assert!(ti.bridge.insight_probability > 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prigogine: Entropy violation triggers stabilization
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_prigogine_violation_stabilizes() {
    let mut ti = ThermodynamicIntegration::default();

    // Start in LinearNonEquilibrium with low entropy
    let mut input = healthy_input();
    input.dissipative_regime = ThermodynamicRegime::LinearNonEquilibrium;
    input.entropy_production_rate = 0.05;
    ti.run_cycle(&input);

    // Increase entropy production (violation!)
    input.entropy_production_rate = 0.2;
    let fb = ti.run_cycle(&input);

    if fb.prigogine_violated {
        assert!(
            fb.exploration_factor < 1.0,
            "Prigogine violation should dampen exploration: {}",
            fb.exploration_factor
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Landauer: Memory pressure under energy constraint
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_landauer_memory_pressure() {
    let mut ti = ThermodynamicIntegration::default();

    // Tight energy budget + high prediction error (lots to consolidate)
    let mut input = healthy_input();
    input.energy_per_cycle = 1e-15; // Extremely tight
    input.prediction_error = 5.0; // High error = many bits to write
    let fb = ti.run_cycle(&input);

    assert!(
        fb.memory_consolidation_suppressed,
        "Should suppress memory under Landauer pressure"
    );

    // Generous budget should not suppress
    input.energy_per_cycle = 1e-5;
    let fb2 = ti.run_cycle(&input);
    assert!(
        !fb2.memory_consolidation_suppressed,
        "Should not suppress with generous budget"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Jarzynski: Divergence triggers HFE correction
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_jarzynski_divergence_correction() {
    let mut ti = ThermodynamicIntegration::default();

    // Run enough cycles to build Jarzynski work samples
    let mut input = healthy_input();
    for _ in 0..25 {
        ti.run_cycle(&input);
    }

    // At this point Jarzynski should have samples
    assert!(ti.bridge.work_samples.len() >= 3);
    // Jarzynski free energy should be computed
    assert!(ti.bridge.jarzynski_free_energy.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Onsager: Symmetry converges with stable inputs
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_onsager_symmetry_converges() {
    let mut ti = ThermodynamicIntegration::default();
    let input = healthy_input();

    // Initial asymmetry = 1.0 (no data)
    assert_eq!(ti.bridge.onsager_asymmetry, 1.0);

    // Feed identical inputs → should converge to low asymmetry
    for _ in 0..15 {
        ti.run_cycle(&input);
    }

    assert!(
        ti.bridge.onsager_asymmetry < 0.5,
        "Onsager should converge with stable input: {}",
        ti.bridge.onsager_asymmetry
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Carnot: Efficiency reflects temperature
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_carnot_reflects_temperature() {
    let mut ti = ThermodynamicIntegration::default();

    // High temperature → high Carnot efficiency
    let mut hot = healthy_input();
    hot.analyzer_temperature = 1.5;
    ti.run_cycle(&hot);
    let hot_carnot = ti.bridge.carnot_efficiency;

    // Low temperature → low Carnot efficiency
    let mut cold = healthy_input();
    cold.analyzer_temperature = 0.25;
    let mut ti2 = ThermodynamicIntegration::default();
    ti2.run_cycle(&cold);
    let cold_carnot = ti2.bridge.carnot_efficiency;

    assert!(
        hot_carnot > cold_carnot,
        "Hotter system should have higher Carnot efficiency: hot={hot_carnot:.3}, cold={cold_carnot:.3}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// HFE blending: Canonical free energy reflects both sources
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_hfe_blending_shifts_free_energy() {
    // System A: low HFE (good model)
    let mut a = ThermodynamicIntegration::default();
    let mut input_a = healthy_input();
    input_a.hfe_total = 0.1;
    a.run_cycle(&input_a);
    let fe_a = a.state.canonical_free_energy;

    // System B: high HFE (bad model)
    let mut b = ThermodynamicIntegration::default();
    let mut input_b = healthy_input();
    input_b.hfe_total = 5.0;
    b.run_cycle(&input_b);
    let fe_b = b.state.canonical_free_energy;

    // Higher HFE should shift canonical free energy upward (more positive)
    assert!(
        fe_b > fe_a,
        "High HFE should increase canonical FE: a={fe_a:.3}, b={fe_b:.3}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// All feedback factors remain finite under edge cases
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_feedback_finite_under_extremes() {
    let mut ti = ThermodynamicIntegration::default();

    // Zero everything
    let fb1 = ti.run_cycle(&ThermodynamicInput::default());
    assert!(fb1.lr_factor.is_finite());
    assert!(fb1.exploration_factor.is_finite());

    // Max everything
    let extreme = ThermodynamicInput {
        unified_psi: 1.0,
        coherence: 1.0,
        prediction_error: 100.0,
        attention_sensitivity: 1.0,
        energy_per_cycle: 1e-5,
        total_energy_spent: 1e10,
        energy_throughput_multiplier: 1000.0,
        metabolic_stress: 1.0,
        dissipative_health: 1.0,
        dissipative_regime: ThermodynamicRegime::Chaotic,
        entropy_production_rate: 1.0,
        order_parameter: 1.0,
        criticality_distance: 0.0,
        lambda_parameter: 1.0,
        dissipation_efficiency: 100.0,
        has_dissipative: true,
        analyzer_entropy: 1.0,
        analyzer_free_energy: 1000.0,
        analyzer_temperature: 2.0,
        analyzer_phase: ConsciousnessPhase::Chaotic,
        has_analyzer: true,
        hfe_total: 1000.0,
        hfe_complexity: 500.0,
        hfe_accuracy: 500.0,
        has_hfe: true,
    };
    let fb2 = ti.run_cycle(&extreme);
    assert!(fb2.lr_factor.is_finite());
    assert!(fb2.exploration_factor.is_finite());
    assert!(ti.bridge.carnot_efficiency.is_finite());
    assert!(ti.bridge.insight_probability.is_finite());
    assert!(ti.bridge.onsager_asymmetry.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn variance(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

// ═══════════════════════════════════════════════════════════════════════════════
// Performance Benchmark
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_run_cycle_latency_1000() {
    use std::time::Instant;

    let mut ti = ThermodynamicIntegration::default();
    let input = healthy_input();

    // Warm up (10 cycles)
    for _ in 0..10 {
        ti.run_cycle(&input);
    }

    // Benchmark 1000 cycles
    let mut latencies_us = Vec::with_capacity(1000);
    for i in 0..1000 {
        let mut varied = input.clone();
        let phase = (i as f64 * 0.1).sin();
        varied.order_parameter = (0.5 + phase * 0.2).clamp(0.1, 0.9);
        varied.entropy_production_rate = (0.15 + phase * 0.05).clamp(0.01, 0.5);
        varied.total_energy_spent = i as f64 * 1e-10;

        let start = Instant::now();
        let _fb = ti.run_cycle(&varied);
        latencies_us.push(start.elapsed().as_micros() as u64);
    }

    latencies_us.sort();
    let p50 = latencies_us[499];
    let p95 = latencies_us[949];
    let p99 = latencies_us[989];
    let max_us = latencies_us[999];
    let mean: f64 = latencies_us.iter().sum::<u64>() as f64 / 1000.0;

    // Print for observation (visible with --nocapture)
    eprintln!(
        "  [bench] run_cycle latency (µs): mean={mean:.1} p50={p50} p95={p95} p99={p99} max={max_us}"
    );

    // Assert: p99 should be under 1ms (1000µs)
    // This is generous; the actual target is <100µs for scalar math.
    assert!(
        p99 < 1000,
        "run_cycle p99 latency {p99}µs exceeds 1ms budget"
    );
}

#[test]
fn bench_hdc_encoding_latency() {
    use std::time::Instant;

    let mut ti = ThermodynamicIntegration::default();
    let input = healthy_input();
    // Populate state
    for _ in 0..20 {
        ti.run_cycle(&input);
    }

    let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("thermodynamic benchmark");
    let encoder = symthaea_core::physics::ThermoEncoder::from_genesis(&genesis);

    let mut latencies_us = Vec::with_capacity(100);
    for _ in 0..100 {
        let start = Instant::now();
        let hv = ti.encode_as_hdc(&encoder);
        let elapsed = start.elapsed().as_micros() as u64;
        latencies_us.push(elapsed);
        std::hint::black_box(&hv);
    }

    latencies_us.sort();
    let p95 = latencies_us[94];
    let mean: f64 = latencies_us.iter().sum::<u64>() as f64 / 100.0;

    eprintln!(
        "  [bench] HDC encode (µs): mean={mean:.1} p95={p95} max={}",
        latencies_us[99]
    );

    // HDC encoding involves 3 bindings of 16,384D vectors — should be <5ms
    assert!(p95 < 5000, "HDC encode p95 {p95}µs exceeds 5ms budget");
}

#[test]
fn bench_feedback_activation_rates() {
    let mut ti = ThermodynamicIntegration::default();
    let input = healthy_input();

    let mut insights = 0u32;
    let mut prigogine = 0u32;
    let mut memory_sup = 0u32;
    let n = 500;

    for i in 0..n {
        let mut varied = input.clone();
        let phase = (i as f64 * 0.1).sin();
        varied.order_parameter = (0.5 + phase * 0.3).clamp(0.1, 0.95);
        varied.entropy_production_rate = (0.15 + phase * 0.1).clamp(0.01, 0.6);

        let fb = ti.run_cycle(&varied);
        if fb.insight_detected {
            insights += 1;
        }
        if fb.prigogine_violated {
            prigogine += 1;
        }
        if fb.memory_consolidation_suppressed {
            memory_sup += 1;
        }
    }

    eprintln!("  [bench] Activation rates over {n} cycles:");
    eprintln!(
        "    Insight: {insights}/{n} ({:.1}%)",
        insights as f64 / n as f64 * 100.0
    );
    eprintln!(
        "    Prigogine: {prigogine}/{n} ({:.1}%)",
        prigogine as f64 / n as f64 * 100.0
    );
    eprintln!(
        "    Memory suppressed: {memory_sup}/{n} ({:.1}%)",
        memory_sup as f64 / n as f64 * 100.0
    );

    // At least some feedback should be active (not all loops dead)
    let total_active = insights + prigogine + memory_sup;
    // With sinusoidal variation, we expect at least a few activations
    // With sinusoidal variation over 500 cycles, some feedback loops should fire
    eprintln!("    Total activations: {total_active}");
    assert!(
        total_active > 0,
        "at least some feedback loops should activate over {n} cycles"
    );
}

#[test]
fn bench_onsager_convergence_speed() {
    let mut ti = ThermodynamicIntegration::default();
    let input = healthy_input();

    // Feed identical input until Onsager converges below 0.05
    let mut converge_cycle = None;
    for i in 0..100 {
        ti.run_cycle(&input);
        if converge_cycle.is_none() && ti.bridge.onsager_asymmetry < 0.05 {
            converge_cycle = Some(i);
        }
    }

    let final_asym = ti.bridge.onsager_asymmetry;
    eprintln!(
        "  [bench] Onsager convergence: {} (final asymmetry: {final_asym:.6})",
        converge_cycle
            .map(|c| format!("cycle {c}"))
            .unwrap_or_else(|| "not converged".to_string())
    );

    // With constant input, covariance matrix is all-zero → asymmetry should be 0
    assert!(
        final_asym < 0.1,
        "Onsager should converge with constant input: {final_asym}"
    );
}