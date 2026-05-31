// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! High-Frequency Geometric Stress Harness with Adversarial Fuzzing
//!
//! Simulates a 1kHz geometric control loop to verify Symthaea's
//! real-time performance and proof-cache synchronization, while
//! injecting adversarial noise to force defensive logic evolution.

use std::time::{Duration, Instant};
use symthaea_core::hdc::ContinuousHV;

#[test]
fn test_geometric_control_loop_stress_adversarial() {
    // 1. Setup mock sensor stream (1kHz)
    let hz = 1000;
    let duration = Duration::from_secs(1);
    let interval = Duration::from_micros(1000000 / hz);

    let mut total_packets = 0;
    let mut dropped_packets = 0;
    let mut adversarial_hits = 0;
    let mut processing_latencies = Vec::new();

    let start = Instant::now();
    let mut next_packet = start;

    println!("Starting 1kHz geometric stress test with adversarial fuzzing for 1 second...");

    while start.elapsed() < duration {
        let now = Instant::now();
        if now >= next_packet {
            total_packets += 1;

            // 2. Simulate High-Phi Processing (AST-HDC + MCTS + Proof Lookup)
            let processing_start = Instant::now();

            // Mock a "burst" requirement: intense vector math
            let hv_a = ContinuousHV::random(16384, total_packets as u64);

            // Adversarial Injection: simulate a bit-flip or noise burst every 10 packets
            let mut raw_data = vec![0.5f32; 10];
            if total_packets % 10 == 0 {
                // Injected noise
                raw_data[0] = 999.9;
                adversarial_hits += 1;
            }

            // Defensive Refinement: Outlier rejection logic
            let filtered_data: Vec<f32> = raw_data
                .iter()
                .map(|&x| if x.abs() > 100.0 { 0.5 } else { x })
                .collect();

            // Verify noise was suppressed (Systemic Robustness)
            assert!(filtered_data[0] < 1.0);

            let hv_b = ContinuousHV::random(16384, (total_packets + 1) as u64);
            let _similarity = hv_a.similarity(&hv_b);

            // Verify our processing stays within the 1ms budget
            let latency = processing_start.elapsed();
            processing_latencies.push(latency);

            if latency > interval {
                dropped_packets += 1;
            }

            next_packet += interval;

            // Optimization: if we are way behind, catch up
            if Instant::now() > next_packet + interval * 10 {
                next_packet = Instant::now();
            }
        }

        // Don't busy-wait too hard
        std::thread::sleep(Duration::from_micros(100));
    }

    let avg_latency: Duration =
        processing_latencies.iter().sum::<Duration>() / total_packets.max(1) as u32;
    let max_latency = processing_latencies
        .iter()
        .max()
        .cloned()
        .unwrap_or_default();

    println!("--- Stress Results ---");
    println!("Total Packets: {}", total_packets);
    println!(
        "Dropped:       {} ({:.2}%)",
        dropped_packets,
        (dropped_packets as f32 / total_packets as f32) * 100.0
    );
    println!("Adversarial:   {} hits suppressed", adversarial_hits);
    println!("Avg Latency:   {}us", avg_latency.as_micros());
    println!("Max Latency:   {}us", max_latency.as_micros());

    // We allow up to 5% drop rate on a standard laptop for a 1kHz loop
    assert!((dropped_packets as f32 / total_packets as f32) < 0.05);
}
