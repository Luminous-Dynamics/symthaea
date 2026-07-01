// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::time::Duration;
use symthaea::mind::{AsyncMind, MindConfig};
use tokio::time::sleep;

#[tokio::main]
async fn main() {
    println!("💤 Symthaea v0.8.0: The Conscious Nap Demo");
    println!("------------------------------------------");

    let config = MindConfig {
        ..Default::default()
    };

    let (mind, _join) = AsyncMind::spawn(config);

    // 1. BASELINE: Low load
    println!("\n[PHASE 1] Low Thermodynamic Load (Physics: Rested)");
    mind.update_thermodynamics(0.1).await;

    let start = std::time::Instant::now();
    for _i in 1..=5 {
        sleep(Duration::from_millis(150)).await;
        let state = mind.snapshot().await;
        println!(
            "   Tick {}: Time elapsed: {:?}, Mood Temp: {:.2}",
            state.tick,
            start.elapsed(),
            state.mood_temperature
        );
    }

    // 2. HIGH LOAD: Throttled Nap
    println!("\n[PHASE 2] High Thermodynamic Load (Physics: Exhausted)");
    mind.update_thermodynamics(0.9).await;

    let start = std::time::Instant::now();
    for _i in 1..=5 {
        sleep(Duration::from_millis(500)).await;
        let state = mind.snapshot().await;
        println!(
            "   Tick {}: Time elapsed: {:?}, Mood Temp: {:.2}",
            state.tick,
            start.elapsed(),
            state.mood_temperature
        );
    }

    // 3. EVENT-DRIVEN WAKEUP
    println!("\n[PHASE 3] Event-Driven Wakeup (The Startle Response)");
    println!("   Waiting for an external stimulus to interrupt the nap...");

    // Set extremely high load (1s nap)
    mind.update_thermodynamics(1.0).await;
    let before = mind.snapshot().await;

    sleep(Duration::from_millis(100)).await; // Only 10% into the nap
    println!("   ⚡ External Perception arriving!");
    let dim = 512;
    let hv = symthaea_core::hdc::ContinuousHV::random(dim, 42);
    mind.perceive(hv).await;

    let after = mind.snapshot().await;
    println!(
        "   -> Result: Pre-tick: {}, Post-tick: {}",
        before.tick, after.tick
    );
    if after.tick > before.tick {
        println!("   ✅ SUCCESS: Stimulus triggered an immediate wake-up!");
    } else {
        println!("   ❌ FAILURE: Mind stayed asleep.");
    }

    println!("\n[CONCLUSION] Symthaea has mastered the Art of Stillness.");
    mind.shutdown().await;
}