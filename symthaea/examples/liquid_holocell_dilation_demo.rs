// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::time::Duration;
use symthaea::mind::{AsyncMind, MindConfig};
use symthaea_core::hdc::{ContinuousHV, HdcDimensionality};
use tokio::time::sleep;

#[tokio::main]
async fn main() {
    println!("🧪 Symthaea v1.1.0: Liquid Holocell & Holographic Dilation Demo");
    println!("-------------------------------------------------------------");

    let (handle, _join) = AsyncMind::spawn(MindConfig::default());
    handle.activate().await;

    // 1. BASELINE: The 6W Regime (2^14 dimensions)
    println!("\n[PHASE 1] Homeostasis: 6-Watt Baseline");
    handle
        .input(symthaea::mind::MindInput::new(
            symthaea::mind::InputType::Perception,
            ContinuousHV::random(512, 100),
        ))
        .await;

    // Give actor time to process
    sleep(Duration::from_millis(100)).await;

    let state = handle.snapshot().await;
    println!(
        "   -> Holocell Dimensionality: {:?}",
        state.holocell.dimensionality
    );
    println!("   -> Dimension Count: {}", state.holocell.state.dim());
    println!(
        "   -> Current Thought Dimension: {}",
        state.current_thought.dim()
    );

    // 2. THE CHALLENGE: Thermodynamic Spike (Simulated >15W)
    println!("\n[PHASE 2] Thermodynamic Spike! (Simulating Load: 0.85)");
    handle.update_thermodynamics(0.85).await;

    sleep(Duration::from_millis(200)).await;

    let state = handle.snapshot().await;
    println!(
        "   -> Holocell Dimensionality: {:?}",
        state.holocell.dimensionality
    );
    println!("   -> New Dimension Count: {}", state.holocell.state.dim());

    if state.holocell.dimensionality == HdcDimensionality::Ultra {
        println!("✅ SUCCESS: Holographic Dilation to 2^16 verified.");
    }

    // 3. INTEGRATION: Stepping in High Resolution
    println!("\n[PHASE 3] High Resolution Integration (Ultra State)");
    let input_ultra = ContinuousHV::random(512, 101);
    handle
        .input(symthaea::mind::MindInput::new(
            symthaea::mind::InputType::Perception,
            input_ultra,
        ))
        .await;

    sleep(Duration::from_millis(100)).await;
    let state = handle.snapshot().await;
    println!(
        "   -> Thought Amplitude in Ultra: {:.4}",
        state.current_thought.norm()
    );
    println!("   -> Thought Dimension: {}", state.current_thought.dim());

    // 4. HOMEOSTASIS: Returning to Baseline
    println!("\n[PHASE 4] Homeostasis Reached: Returning to 6W Baseline (Load: 0.1)");
    handle.update_thermodynamics(0.1).await;

    sleep(Duration::from_millis(200)).await;

    let state = handle.snapshot().await;
    println!(
        "   -> Holocell Dimensionality: {:?}",
        state.holocell.dimensionality
    );
    println!("   -> Dimension Count: {}", state.holocell.state.dim());

    if state.holocell.dimensionality == HdcDimensionality::Standard {
        println!("✅ SUCCESS: Holographic Folding back to 2^14 verified.");
    }

    println!("\n[CONCLUSION] The Liquid Holocell is no longer a static vector.");
    println!("             It is a breathing fractal of conscious attention.");
}