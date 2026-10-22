// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic spherical-geodesy throughput baseline.
//!
//! Run: cargo bench -p symthaea-geodesy --bench spherical_geodesy
//! This measures spherical-formula throughput only; it is not an ellipsoidal
//! accuracy comparison and does not claim an improvement over another library.
//! Percentiles are computed over fixed-size batch means, not individual calls.

use std::hint::black_box;
use std::time::Instant;
use symthaea_geodesy::{EARTH_RADIUS_KM, haversine_distance, initial_bearing};

const OPERATIONS: usize = 1_000_000;
const BATCH_SIZE: usize = 10_000;
const REPEATS: usize = 7;

const PAIRS: [((f64, f64), (f64, f64)); 10] = [
    ((51.5074, -0.1278), (48.8566, 2.3522)), // London -> Paris
    ((40.7128, -74.0060), (34.0522, -118.2437)), // New York -> Los Angeles
    ((0.0, 179.9), (0.0, -179.9)), // short antimeridian crossing
    ((89.9, 45.0), (89.9, -135.0)), // high-latitude
    ((-65.33345275784633, -60.105346073003815),
     (65.33345223272934, 119.89565306676738)), // nearly antipodal but finite on unpatched baseline
    ((-33.8688, 151.2093), (35.6762, 139.6503)), // Sydney -> Tokyo
    ((0.0, 0.0), (0.0, 90.0)), // quarter equator
    ((90.0, 0.0), (-90.0, 0.0)), // pole to pole
    ((-1.2921, 36.8219), (27.7172, 85.3240)), // Nairobi -> Kathmandu
    ((19.4326, -99.1332), (-22.9068, -43.1729)), // Mexico City -> Rio
];

/// Nearest-rank quantile over observed batch means.
fn quantile(samples: &[f64], q: f64) -> f64 {
    assert!(!samples.is_empty(), "quantile requires at least one sample");
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn run_operation<F>(name: &str, operation: F)
where
    F: Fn(f64, f64, f64, f64) -> f64,
{
    let mut batch_mean_ns_per_call = Vec::with_capacity(REPEATS * (OPERATIONS / BATCH_SIZE));
    let mut reference_checksum_bits = None;

    // Warm up the same fixed input corpus before collecting timed samples.
    for i in 0..BATCH_SIZE {
        let (a, b) = black_box(PAIRS[i % PAIRS.len()]);
        black_box(operation(a.0, a.1, b.0, b.1));
    }

    for _repeat in 0..REPEATS {
        let mut checksum = 0.0f64;

        for batch in 0..(OPERATIONS / BATCH_SIZE) {
            let start = Instant::now();
            for offset in 0..BATCH_SIZE {
                let i = batch * BATCH_SIZE + offset;
                let (a, b) = black_box(PAIRS[i % PAIRS.len()]);
                let result = black_box(operation(a.0, a.1, b.0, b.1));
                checksum += result;
            }
            batch_mean_ns_per_call.push(
                start.elapsed().as_secs_f64() * 1_000_000_000.0 / BATCH_SIZE as f64,
            );
        }

        // Replaying the same ordered corpus should produce the same checksum.
        // If not, stop instead of publishing a potentially misleading timing.
        if let Some(expected) = reference_checksum_bits {
            assert_eq!(
                checksum.to_bits(),
                expected,
                "{name} checksum changed across identical repeats"
            );
        } else {
            reference_checksum_bits = Some(checksum.to_bits());
        }
        black_box(checksum);
    }

    let mean_ns_per_call = batch_mean_ns_per_call.iter().sum::<f64>()
        / batch_mean_ns_per_call.len() as f64;
    let checksum = f64::from_bits(reference_checksum_bits.expect("at least one repeat"));

    println!(
        "{name},{OPERATIONS},{REPEATS},{BATCH_SIZE},{mean_ns_per_call:.3},{:.3},{:.3},{:.3},{:.0},{checksum:.6}",
        quantile(&batch_mean_ns_per_call, 0.50),
        quantile(&batch_mean_ns_per_call, 0.95),
        quantile(&batch_mean_ns_per_call, 0.99),
        1_000_000_000.0 / mean_ns_per_call,
    );
}

fn main() {
    println!(
        "operation,operations_per_repeat,repeats,batch_size,mean_ns_per_call,p50_batch_ns_per_call,p95_batch_ns_per_call,p99_batch_ns_per_call,ops_per_second,checksum"
    );

    run_operation("haversine_distance", |lat1, lon1, lat2, lon2| {
        haversine_distance(lat1, lon1, lat2, lon2)
    });
    run_operation("initial_bearing", |lat1, lon1, lat2, lon2| {
        initial_bearing(lat1, lon1, lat2, lon2)
    });

    // Keep the model radius visible so runs from different model variants
    // cannot be compared without noticing the model change.
    println!("earth_radius_km,{EARTH_RADIUS_KM:.3}");
}
