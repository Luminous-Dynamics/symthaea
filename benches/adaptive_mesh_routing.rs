// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dependency-free benchmark for the current mesh router.
//!
//! Run: cargo bench --bench adaptive_mesh_routing --features mesh
//! Reports batch-mean latency distributions for policy CPU cost only. This
//! does not model end-to-end packet delivery, link dynamics, or adversarial
//! convergence. Compare exact commits on the same host/toolchain.

use std::hint::black_box;
use std::time::Instant;
use symthaea::cognitive_loop::{
    ConsciousRoutingDecision, ConsciousnessAwareRouter, PayloadClass, PayloadClassifier,
};

const REPEATS: usize = 7;
const ROUTE_ITERATIONS: usize = 200_000;
const ROUTE_BATCH_SIZE: usize = 5_000;
const UPDATE_ROUNDS_PER_REPEAT: usize = 250;

fn peer_id(index: usize) -> [u8; 8] {
    (index as u64).to_le_bytes()
}

/// Nearest-rank quantile over observed batch means.
fn quantile(samples: &[f64], q: f64) -> f64 {
    assert!(!samples.is_empty(), "quantile requires at least one sample");
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn make_router(peer_count: usize) -> ConsciousnessAwareRouter {
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
    router
}

fn run_route_case(peer_count: usize) {
    let mut router = make_router(peer_count);
    let classifier = PayloadClassifier::default();

    // Warm up with the same operation mix before recording timed batches.
    for i in 0..ROUTE_BATCH_SIZE {
        let class = match i % 4 {
            0 => PayloadClass::Emergency,
            1 => PayloadClass::ConsciousnessDelta,
            _ => PayloadClass::Discovery,
        };
        black_box(router.route(
            class,
            black_box(40 + (i % 2_000)),
            (i % 3) as u8,
            &classifier,
        ));
    }

    let mut batch_mean_us_per_route =
        Vec::with_capacity(REPEATS * (ROUTE_ITERATIONS / ROUTE_BATCH_SIZE));
    let mut expected_normal_decisions = None;

    for _repeat in 0..REPEATS {
        let mut normal_decisions = 0usize;
        for batch_start in (0..ROUTE_ITERATIONS).step_by(ROUTE_BATCH_SIZE) {
            let start = Instant::now();
            for offset in 0..ROUTE_BATCH_SIZE {
                let i = batch_start + offset;
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
                    normal_decisions = normal_decisions.wrapping_add(1);
                }
            }
            batch_mean_us_per_route.push(
                start.elapsed().as_secs_f64() * 1_000_000.0 / ROUTE_BATCH_SIZE as f64,
            );
        }

        if let Some(expected) = expected_normal_decisions {
            assert_eq!(
                normal_decisions, expected,
                "route decision count changed across identical repeats"
            );
        } else {
            expected_normal_decisions = Some(normal_decisions);
        }
        black_box(normal_decisions);
    }

    let mean_us = batch_mean_us_per_route.iter().sum::<f64>()
        / batch_mean_us_per_route.len() as f64;
    println!(
        "route,{peer_count},{ROUTE_ITERATIONS},{REPEATS},{ROUTE_BATCH_SIZE},{mean_us:.6},{:.6},{:.6},{:.6},{:.0},{}",
        quantile(&batch_mean_us_per_route, 0.50),
        quantile(&batch_mean_us_per_route, 0.95),
        quantile(&batch_mean_us_per_route, 0.99),
        1_000_000.0 / mean_us,
        expected_normal_decisions.expect("at least one repeat"),
    );
}

fn run_peer_update_case(peer_count: usize) {
    let mut router = make_router(peer_count);
    let mut batch_mean_us_per_update =
        Vec::with_capacity(REPEATS * UPDATE_ROUNDS_PER_REPEAT);

    for repeat in 0..REPEATS {
        for round in 0..UPDATE_ROUNDS_PER_REPEAT {
            // Monotonic cycles avoid benchmarking updates rejected as stale.
            let cycle = (repeat * UPDATE_ROUNDS_PER_REPEAT + round + 2) as u64;
            let start = Instant::now();
            for i in 0..peer_count {
                let phi = ((i * 31 + cycle as usize * 7) % 100) as f32 / 100.0;
                let level = ((i * 11 + cycle as usize * 3) % 100) as f32 / 100.0;
                router.update_peer(peer_id(i), phi, level, (i % 5) as u8, cycle);
            }
            batch_mean_us_per_update.push(
                start.elapsed().as_secs_f64() * 1_000_000.0 / peer_count as f64,
            );
        }
    }

    black_box(router.collective_phi());
    black_box(router.peer_count());
    let mean_us = batch_mean_us_per_update.iter().sum::<f64>()
        / batch_mean_us_per_update.len() as f64;
    println!(
        "peer_update,{peer_count},{},{REPEATS},{peer_count},{mean_us:.6},{:.6},{:.6},{:.6},{:.0},{}",
        peer_count * UPDATE_ROUNDS_PER_REPEAT,
        quantile(&batch_mean_us_per_update, 0.50),
        quantile(&batch_mean_us_per_update, 0.95),
        quantile(&batch_mean_us_per_update, 0.99),
        1_000_000.0 / mean_us,
        router.peer_count(),
    );
}

fn main() {
    println!(
        "workload,peers,operations_per_repeat,repeats,batch_size,mean_us_per_operation,p50_batch_us_per_operation,p95_batch_us_per_operation,p99_batch_us_per_operation,operations_per_second,checksum_or_peer_count"
    );
    for peer_count in [16, 64, 256] {
        run_route_case(peer_count);
    }
    for peer_count in [16, 64, 256] {
        run_peer_update_case(peer_count);
    }
}
