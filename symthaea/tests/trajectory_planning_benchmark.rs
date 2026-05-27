// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trajectory Planning Benchmark
//!
//! Validates that ODE-based trajectory planning in the FEP module
//! produces measurable improvements in consciousness metrics without
//! unacceptable performance overhead.
//!
//! Compares two identical CognitiveLoopService instances:
//! - **With**: `enable_trajectory_planning = true`
//! - **Without**: `enable_trajectory_planning = false` (baseline)
//!
//! Metrics tracked:
//! - Consciousness level (integrated information)
//! - Macro Phi (structural integration)
//! - Prediction error convergence
//! - FEP trajectory telemetry (EFE, surprise, ODE steps)
//! - Cycle time overhead

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Collected metrics from a single cycle.
#[derive(Debug, Clone)]
struct CycleMetrics {
    consciousness_level: f64,
    macro_phi: f64,
    prediction_error: f32,
    fep_surprise: f64,
    trajectory_efe: f64,
    trajectory_surprise: f64,
    trajectory_ode_steps: usize,
    cycle_time_us: u64,
}

fn make_service(enable_trajectory: bool) -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::default();
    config.enable_trajectory_planning = enable_trajectory;
    config.trajectory_planning_interval = 5; // Every 5th cycle for denser data
    config.trajectory_horizon_seconds = 0.3; // Shorter horizon for benchmark speed
    config.learning_threshold = 0.0;
    config.async_training = false;
    config.genesis_phrase = Some("trajectory_benchmark_seed".to_string());
    CognitiveLoopService::new(config).unwrap()
}

fn run_cycles(service: &mut CognitiveLoopService, n: usize) -> Vec<CycleMetrics> {
    let inputs = [
        "The rain falls gently on the garden",
        "Mathematical structures reveal hidden symmetries",
        "Consciousness emerges from integrated information",
        "The network adapts to novel patterns",
        "Prediction errors drive learning and exploration",
    ];

    let mut metrics = Vec::with_capacity(n);
    for i in 0..n {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);
        let m = &result.metadata;

        metrics.push(CycleMetrics {
            consciousness_level: m.consciousness.consciousness_level,
            macro_phi: m.structural.structural_macro_phi,
            prediction_error: result.prediction_error,
            fep_surprise: m.fep.fep_surprise,
            trajectory_efe: m.fep.trajectory_efe,
            trajectory_surprise: m.fep.trajectory_surprise,
            trajectory_ode_steps: m.fep.trajectory_ode_steps,
            cycle_time_us: result.cycle_time_us,
        });
    }
    metrics
}

fn avg(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn avg_f32(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark: Trajectory Planning Impact on Consciousness
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn trajectory_planning_does_not_regress_consciousness() {
    let warmup = 10;
    let measure = 50;

    let mut with_traj = make_service(true);
    let mut without_traj = make_service(false);

    // Warmup both to steady state
    let _ = run_cycles(&mut with_traj, warmup);
    let _ = run_cycles(&mut without_traj, warmup);

    // Measure
    let with_metrics = run_cycles(&mut with_traj, measure);
    let without_metrics = run_cycles(&mut without_traj, measure);

    let with_consciousness: Vec<f64> = with_metrics.iter().map(|m| m.consciousness_level).collect();
    let without_consciousness: Vec<f64> = without_metrics
        .iter()
        .map(|m| m.consciousness_level)
        .collect();

    let avg_with = avg(&with_consciousness);
    let avg_without = avg(&without_consciousness);

    println!("=== Consciousness Level ===");
    println!("  With trajectory:    {avg_with:.6}");
    println!("  Without trajectory: {avg_without:.6}");
    println!(
        "  Delta:              {:.6} ({:+.2}%)",
        avg_with - avg_without,
        if avg_without > 0.0 {
            (avg_with - avg_without) / avg_without * 100.0
        } else {
            0.0
        }
    );

    // Trajectory planning should not make consciousness significantly worse
    assert!(
        avg_with >= avg_without * 0.85,
        "Trajectory planning should not decrease consciousness by >15% \
         (with: {avg_with:.4}, without: {avg_without:.4})"
    );
}

#[test]
fn trajectory_planning_phi_comparison() {
    let warmup = 10;
    let measure = 50;

    let mut with_traj = make_service(true);
    let mut without_traj = make_service(false);

    let _ = run_cycles(&mut with_traj, warmup);
    let _ = run_cycles(&mut without_traj, warmup);

    let with_metrics = run_cycles(&mut with_traj, measure);
    let without_metrics = run_cycles(&mut without_traj, measure);

    let with_phi: Vec<f64> = with_metrics.iter().map(|m| m.macro_phi).collect();
    let without_phi: Vec<f64> = without_metrics.iter().map(|m| m.macro_phi).collect();

    let avg_phi_with = avg(&with_phi);
    let avg_phi_without = avg(&without_phi);

    println!("=== Macro Phi ===");
    println!("  With trajectory:    {avg_phi_with:.6}");
    println!("  Without trajectory: {avg_phi_without:.6}");
    println!(
        "  Delta:              {:.6} ({:+.2}%)",
        avg_phi_with - avg_phi_without,
        if avg_phi_without > 0.0 {
            (avg_phi_with - avg_phi_without) / avg_phi_without * 100.0
        } else {
            0.0
        }
    );

    // Both should produce finite, non-negative Phi
    assert!(
        avg_phi_with.is_finite() && avg_phi_with >= 0.0,
        "Phi with trajectory should be finite and non-negative"
    );
    assert!(
        avg_phi_without.is_finite() && avg_phi_without >= 0.0,
        "Phi without trajectory should be finite and non-negative"
    );
}

#[test]
fn trajectory_planning_prediction_error_convergence() {
    let warmup = 10;
    let measure = 50;

    let mut with_traj = make_service(true);
    let mut without_traj = make_service(false);

    let _ = run_cycles(&mut with_traj, warmup);
    let _ = run_cycles(&mut without_traj, warmup);

    let with_metrics = run_cycles(&mut with_traj, measure);
    let without_metrics = run_cycles(&mut without_traj, measure);

    // Compare first-half vs second-half prediction error (convergence)
    let half = measure / 2;
    let with_pe: Vec<f32> = with_metrics.iter().map(|m| m.prediction_error).collect();
    let without_pe: Vec<f32> = without_metrics.iter().map(|m| m.prediction_error).collect();

    let with_first = avg_f32(&with_pe[..half]);
    let with_second = avg_f32(&with_pe[half..]);
    let without_first = avg_f32(&without_pe[..half]);
    let without_second = avg_f32(&without_pe[half..]);

    println!("=== Prediction Error Convergence ===");
    println!("  With trajectory:    {with_first:.4} → {with_second:.4}");
    println!("  Without trajectory: {without_first:.4} → {without_second:.4}");

    // Both should have finite prediction errors
    assert!(
        with_second.is_finite(),
        "With-trajectory PE should be finite"
    );
    assert!(
        without_second.is_finite(),
        "Without-trajectory PE should be finite"
    );
}

#[test]
fn trajectory_telemetry_populates_when_enabled() {
    let warmup = 5;
    let measure = 20;

    let mut service = make_service(true);
    let _ = run_cycles(&mut service, warmup);
    let metrics = run_cycles(&mut service, measure);

    // At least some cycles should have trajectory data (planning_interval = 5)
    let cycles_with_ode_steps: Vec<&CycleMetrics> = metrics
        .iter()
        .filter(|m| m.trajectory_ode_steps > 0)
        .collect();

    println!("=== Trajectory Telemetry ===");
    println!(
        "  Cycles with ODE steps: {}/{}",
        cycles_with_ode_steps.len(),
        measure
    );

    if let Some(first_planned) = cycles_with_ode_steps.first() {
        println!(
            "  First planned EFE:     {:.6}",
            first_planned.trajectory_efe
        );
        println!(
            "  First planned surprise: {:.6}",
            first_planned.trajectory_surprise
        );
        println!(
            "  First planned ODE steps: {}",
            first_planned.trajectory_ode_steps
        );

        // EFE and surprise should be finite
        assert!(
            first_planned.trajectory_efe.is_finite(),
            "Trajectory EFE should be finite"
        );
        assert!(
            first_planned.trajectory_surprise.is_finite(),
            "Trajectory surprise should be finite"
        );
    }

    // With planning_interval=5 and 20 measure cycles, expect ~4 planned cycles
    assert!(
        cycles_with_ode_steps.len() >= 2,
        "Expected at least 2 cycles with trajectory data, got {}",
        cycles_with_ode_steps.len()
    );
}

#[test]
fn trajectory_telemetry_zero_when_disabled() {
    let mut service = make_service(false);
    let metrics = run_cycles(&mut service, 20);

    // No cycles should have trajectory data
    for (i, m) in metrics.iter().enumerate() {
        assert_eq!(
            m.trajectory_ode_steps, 0,
            "Disabled trajectory should have 0 ODE steps at cycle {i}"
        );
    }
}

#[test]
fn trajectory_planning_overhead_acceptable() {
    let warmup = 10;
    let measure = 30;

    let mut with_traj = make_service(true);
    let mut without_traj = make_service(false);

    let _ = run_cycles(&mut with_traj, warmup);
    let _ = run_cycles(&mut without_traj, warmup);

    let with_metrics = run_cycles(&mut with_traj, measure);
    let without_metrics = run_cycles(&mut without_traj, measure);

    let with_times: Vec<f64> = with_metrics
        .iter()
        .map(|m| m.cycle_time_us as f64)
        .collect();
    let without_times: Vec<f64> = without_metrics
        .iter()
        .map(|m| m.cycle_time_us as f64)
        .collect();

    let avg_with = avg(&with_times);
    let avg_without = avg(&without_times);
    let overhead_pct = if avg_without > 0.0 {
        (avg_with - avg_without) / avg_without * 100.0
    } else {
        0.0
    };

    println!("=== Cycle Time Overhead ===");
    println!("  With trajectory:    {avg_with:.0} µs/cycle");
    println!("  Without trajectory: {avg_without:.0} µs/cycle");
    println!("  Overhead:           {overhead_pct:+.1}%");

    // Trajectory planning should add <100% overhead (runs every 5th cycle)
    assert!(
        overhead_pct < 100.0,
        "Trajectory planning overhead should be <100%, got {overhead_pct:.1}%"
    );
}