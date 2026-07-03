// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MuJoCo Articulated Benchmark (64-DOF Flagship Humanoid)
//!
//! Exports the 64-DOF morphology and executes a multibody dynamics
//! validation pass using the MuJoCo bridge.

use anyhow::Result;
use symthaea_mujoco_bridge::MuJoCoBridge;
use symthaea_sim_bridge::{EngineeringDomain, SimulationBackend, SimulationRequest, SolverKind};
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🏗️  INITIATING MUJOCO ARTICULATED BENCHMARK...");

    // 1. Initialize MuJoCo Bridge (dry_run for CI/env safety, but exports MJCF)
    let mut bridge = MuJoCoBridge::default();
    bridge.dry_run = true; // Stay in dry-run mode for now to verify MJCF export only
    bridge.asset_dir = "assets/mujoco_bench".to_string();

    info!(
        "📦 Exporting 64-DOF Flagship MJCF to: {}...",
        bridge.asset_dir
    );
    let xml_path = bridge
        .export_flagship_mjcf()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    info!("✅ MJCF Export SUCCESS: {}", xml_path);

    // 2. Prepare Simulation Request
    let request = SimulationRequest::new(
        "flagship-mbd-01",
        EngineeringDomain::Robotics,
        SolverKind::MultibodyDynamics,
        "Validate 64-DOF flagship humanoid articulated gait feasibility.",
    );

    info!("🚀 Dispatching Multibody Dynamics request to MuJoCo solver...");

    // 3. Run Validation
    let result = bridge
        .run(&request)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    info!("✨ BENCHMARK PASS COMPLETE.");
    info!("📊 VALIDATION REPORT:");
    info!("   🆔 Request ID: {}", result.request_id);
    info!("   🎯 Confidence: {:.2}%", result.confidence * 100.0);

    for metric in &result.metrics {
        info!(
            "   🔹 Metric [{}]: {:.4} {}",
            metric.name, metric.value, metric.unit
        );
    }

    if result.converged {
        info!("   ✅ 64-DOF Articulation FEASIBLE.");
    } else {
        tracing::warn!("   ⚠️  64-DOF Articulation FAILED FEASIBILITY CHECK.");
    }

    Ok(())
}
