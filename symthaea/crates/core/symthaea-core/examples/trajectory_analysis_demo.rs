// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trajectory Analysis Demo
//!
//! Demonstrates how design trajectories can be analyzed using temporal binding
//! to measure how coherently a design evolves through operational states.
//!
//! Run with: cargo run --example trajectory_analysis_demo --release

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{
    CoupledPhysicsEngine, FusionReaction, OperatingConditions, TrajectoryAnalysisEngine,
};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           TRAJECTORY ANALYSIS DEMO                                   ║");
    println!("║           Temporal Binding for Design Evolution                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let genesis = GenesisSeed::from_phrase("Trajectory Analysis 2024");
    let physics = CoupledPhysicsEngine::from_genesis(&genesis);

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO 1: POWER RAMP-UP TRAJECTORY
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SCENARIO 1: POWER RAMP-UP (1 kW → 20 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut rampup_trajectory = TrajectoryAnalysisEngine::from_genesis(&genesis);

    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│  Step  Power    State I   Binding   Continuity   Stream Health       │");
    println!("├────────────────────────────────────────────────────────────────────────┤");

    for i in 0..20 {
        let power = 1.0 + i as f64;
        let conditions = OperatingConditions {
            power_kw: power,
            reaction: FusionReaction::DD,
            ..OperatingConditions::consumer()
        };
        let result = physics.simulate(&conditions);
        let state = rampup_trajectory.process(&result);

        if i % 4 == 0 || i == 19 {
            let health = rampup_trajectory.stream_health();
            println!(
                "│  {:>4}  {:>5.0}    {:>6.4}   {:>6.4}    {:>6.4}      {}  │",
                state.step,
                power,
                state.metrics.overall_integration,
                state.temporal_binding,
                state.continuity,
                if health.is_flowing {
                    "FLOWING"
                } else {
                    "BUILDING"
                }
            );
        }
    }

    println!("└────────────────────────────────────────────────────────────────────────┘");

    let rampup_metrics = rampup_trajectory.trajectory_metrics();
    println!("\n  {}", rampup_metrics.summary());
    println!(
        "  Integration trend: {:.4} (negative = declining as power increases)",
        rampup_trajectory.integration_trend()
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO 2: STEADY-STATE OPERATION
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SCENARIO 2: STEADY-STATE OPERATION (5 kW constant)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut steady_trajectory = TrajectoryAnalysisEngine::from_genesis(&genesis);

    for _ in 0..20 {
        let result = physics.simulate(&OperatingConditions::consumer());
        steady_trajectory.process(&result);
    }

    let steady_metrics = steady_trajectory.trajectory_metrics();
    let steady_health = steady_trajectory.stream_health();

    println!("  {}", steady_metrics.summary());
    println!("  Stream: {}", steady_health);
    println!(
        "  Integration trend: {:.4}",
        steady_trajectory.integration_trend()
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO 3: OSCILLATING OPERATION
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SCENARIO 3: OSCILLATING POWER (3-7 kW sinusoidal)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut oscillating_trajectory = TrajectoryAnalysisEngine::from_genesis(&genesis);

    for i in 0..20 {
        let power = 5.0 + 2.0 * (i as f64 * 0.5).sin();
        let conditions = OperatingConditions {
            power_kw: power,
            ..OperatingConditions::consumer()
        };
        let result = physics.simulate(&conditions);
        oscillating_trajectory.process(&result);
    }

    let oscillating_metrics = oscillating_trajectory.trajectory_metrics();
    println!("  {}", oscillating_metrics.summary());
    println!(
        "  Integration trend: {:.4}",
        oscillating_trajectory.integration_trend()
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPARISON SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TRAJECTORY COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│  Scenario      Quality   Coherence   Mean I    Peak I    Valley I     │");
    println!("├────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Ramp-up      {:>6.4}    {:>6.4}     {:>6.4}   {:>6.4}   {:>6.4}      │",
        rampup_metrics.trajectory_quality,
        rampup_metrics.coherence,
        rampup_metrics.mean_integration,
        rampup_metrics.peak_integration,
        rampup_metrics.valley_integration
    );
    println!(
        "│  Steady       {:>6.4}    {:>6.4}     {:>6.4}   {:>6.4}   {:>6.4}      │",
        steady_metrics.trajectory_quality,
        steady_metrics.coherence,
        steady_metrics.mean_integration,
        steady_metrics.peak_integration,
        steady_metrics.valley_integration
    );
    println!(
        "│  Oscillating  {:>6.4}    {:>6.4}     {:>6.4}   {:>6.4}   {:>6.4}      │",
        oscillating_metrics.trajectory_quality,
        oscillating_metrics.coherence,
        oscillating_metrics.mean_integration,
        oscillating_metrics.peak_integration,
        oscillating_metrics.valley_integration
    );
    println!("└────────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // REACTION TYPE COMPARISON
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  REACTION TYPE TRAJECTORIES (20 steps at 5 kW)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let reactions = [
        (FusionReaction::DD, "D-D (2.45 MeV)"),
        (FusionReaction::DT, "D-T (14.1 MeV)"),
        (FusionReaction::DHe3, "D-He3 (aneutronic)"),
    ];

    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│  Reaction          Quality       Coherence   Mean State I   Health    │");
    println!("├────────────────────────────────────────────────────────────────────────┤");

    for (reaction, name) in &reactions {
        let mut traj = TrajectoryAnalysisEngine::from_genesis(&genesis);

        for _ in 0..20 {
            let conditions = OperatingConditions {
                power_kw: 5.0,
                reaction: *reaction,
                ..OperatingConditions::consumer()
            };
            let result = physics.simulate(&conditions);
            traj.process(&result);
        }

        let metrics = traj.trajectory_metrics();
        let health_str = if metrics.is_healthy() {
            "HEALTHY"
        } else {
            "WARNING"
        };

        println!(
            "│  {:18} {:>10.4}    {:>8.4}    {:>10.4}   {:>7}  │",
            name,
            metrics.trajectory_quality,
            metrics.coherence,
            metrics.mean_integration,
            health_str
        );
    }

    println!("└────────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // INTERPRETATION
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  KEY FINDINGS                                                       │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  • Steady-state operation shows highest trajectory coherence        │");
    println!("│  • Ramp-up trajectory shows declining integration (power effect)    │");
    println!("│  • Oscillating trajectory maintains moderate stability              │");
    println!("│  • Trajectory quality combines coherence + state metrics            │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  ENGINEERING IMPLICATIONS                                           │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  • High trajectory coherence → predictable operational behavior     │");
    println!("│  • Stream health indicates temporal integration quality             │");
    println!("│  • Integration trend reveals optimization direction                 │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════\n");
}
