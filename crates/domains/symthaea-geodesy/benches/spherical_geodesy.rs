// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic spherical-geodesy throughput baseline.
//!
//! Run: cargo bench -p symthaea-geodesy --bench spherical_geodesy
//! This measures the current spherical formulas; it is not an ellipsoidal
//! accuracy comparison and does not claim an improvement over another library.

use std::hint::black_box;
use std::time::Instant;
use symthaea_geodesy::{EARTH_RADIUS_KM, haversine_distance, initial_bearing};

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

fn main() {
    let operations = 1_000_000usize;
    println!("operation,operations,elapsed_ms,ops_per_second,checksum");

    let start = Instant::now();
    let mut distance_checksum = 0.0f64;
    for i in 0..operations {
        let (a, b) = black_box(PAIRS[i % PAIRS.len()]);
        let distance = black_box(haversine_distance(a.0, a.1, b.0, b.1));
        distance_checksum += distance;
    }
    let elapsed = start.elapsed();
    black_box(distance_checksum);
    println!(
        "haversine_distance,{operations},{:.3},{:.0},{:.6}",
        elapsed.as_secs_f64() * 1_000.0,
        operations as f64 / elapsed.as_secs_f64(),
        distance_checksum
    );

    let start = Instant::now();
    let mut bearing_checksum = 0.0f64;
    for i in 0..operations {
        let (a, b) = black_box(PAIRS[i % PAIRS.len()]);
        let bearing = black_box(initial_bearing(a.0, a.1, b.0, b.1));
        bearing_checksum += bearing;
    }
    let elapsed = start.elapsed();
    black_box(bearing_checksum);
    println!(
        "initial_bearing,{operations},{:.3},{:.0},{:.6}",
        elapsed.as_secs_f64() * 1_000.0,
        operations as f64 / elapsed.as_secs_f64(),
        bearing_checksum
    );

    // Keep the model radius visible in the output so the benchmark can be
    // compared with other runs that intentionally change the spherical model.
    println!("earth_radius_km,{EARTH_RADIUS_KM:.3}");
}
