// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic microbenchmark for the existing mesh router.
//!
//! Run: cargo bench --bench adaptive_mesh_routing --features mesh
//! Measures policy throughput only; it does not model end-to-end packet delivery.

use std::hint::black_box;
use std::time::Instant;
use symthaea::cognitive_loop::{
    ConsciousRoutingDecision, ConsciousnessAwareRouter, PayloadClass, PayloadClassifier,
};

fn peer_id(index: usize) -> [u8; 8] {
    (index as u64).to_le_bytes()
}

fn run_route_case(peer_count: usize, iterations: usize) {
    let mut router = ConsciousnessAwareRouter::default();
    for i in 0..peer_count {
        router.update_peer(
            peer_id(i),
            (i % 100) as f32 / 100.0,
            ((i * 17) % 100) as f32 / 100.0,
            (i % 5) as u8,
            1,
        );
    }
    let classifier = PayloadClassifier::default();

    let start = Instant::now();
    let mut decisions = 0usize;
    for i in 0..iterations {
        let class = match i % 4 {
            0 => PayloadClass::Emergency,
            1 => PayloadClass::ConsciousnessDelta,
            _ => PayloadClass::Discovery,
        };
        let decision = black_box(router.route(
            class,
            black_box(40 + (i % 2_000)),
            (i % 3) as u8,
            &classifier,
        ));
        if matches!(decision, ConsciousRoutingDecision::Normal(_)) {
            decisions = decisions.wrapping_add(1);
        }
    }
    let elapsed = start.elapsed();
    black_box(decisions);
    let mean_us_per_route = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    println!(
        "route,{peer_count},{iterations},{:.3},{:.6},{decisions}",
        elapsed.as_secs_f64() * 1_000.0,
        mean_us_per_route
    );
}

fn run_peer_update_case(peer_count: usize, rounds: usize) {
    let mut router = ConsciousnessAwareRouter::default();
    for i in 0..peer_count {
        router.update_peer(
            peer_id(i),
            (i % 100) as f32 / 100.0,
            ((i * 17) % 100) as f32 / 100.0,
            (i % 5) as u8,
            1,
        );
    }

    let start = Instant::now();
    for round in 2..(rounds + 2) {
        for i in 0..peer_count {
            // Deterministic bounded changes; monotonic cycle values avoid
            // accidentally benchmarking rejected stale reports.
            let phi = ((i * 31 + round * 7) % 100) as f32 / 100.0;
            let level = ((i * 11 + round * 3) % 100) as f32 / 100.0;
            router.update_peer(peer_id(i), phi, level, (i % 5) as u8, round as u64);
        }
    }
    let elapsed = start.elapsed();
    black_box(router.collective_phi());
    let updates = peer_count * rounds;
    println!(
        "peer_update,{peer_count},{updates},{:.3},{:.3},{}",
        elapsed.as_secs_f64() * 1_000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / updates as f64,
        router.peer_count()
    );
}

fn main() {
    println!("workload,peers,operations,elapsed_ms,mean_us_per_operation,checksum_or_peer_count");
    for peer_count in [16, 64, 256] {
        run_route_case(peer_count, 200_000);
    }
    for peer_count in [16, 64, 256] {
        run_peer_update_case(peer_count, 250);
    }
}
