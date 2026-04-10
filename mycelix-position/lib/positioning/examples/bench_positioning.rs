// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Positioning benchmarks: trilateration, EKF, ranging, GDOP.

use positioning::*;
use std::time::Instant;

fn main() {
    println!("=== POSITIONING BENCHMARKS ===\n");

    bench_trilateration_3d();
    bench_trilateration_2d();
    bench_ekf_convergence();
    bench_pdr_walkabout();
    bench_ranging_models();
    bench_gdop();
    bench_geodetic_roundtrip();

    println!("\n=== COMPLETE ===");
}

fn bench_trilateration_3d() {
    let target: [f64; 3] = [50.0, 30.0, -20.0];
    let anchors: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 100.0],
        [100.0, 100.0, 0.0],
        [50.0, 50.0, 100.0],
    ];
    let ranges: Vec<f64> = anchors
        .iter()
        .map(|a| {
            let dx: f64 = a[0] - target[0];
            let dy: f64 = a[1] - target[1];
            let dz: f64 = a[2] - target[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();
    let sigmas = vec![1.0; 6];
    let trusts = vec![1.0; 6];

    let n = 10_000;
    let start = Instant::now();
    for _ in 0..n {
        let _ = trilaterate_3d(&anchors, &ranges, &sigmas, &trusts);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / n;

    let est = trilaterate_3d(&anchors, &ranges, &sigmas, &trusts).unwrap();
    let error = ((est.position[0] - target[0]).powi(2)
        + (est.position[1] - target[1]).powi(2)
        + (est.position[2] - target[2]).powi(2))
    .sqrt();

    println!("3D Trilateration (6 anchors):");
    println!("  {n} iterations: {elapsed:?} ({per_call:?}/call)");
    println!("  Position error: {error:.6} m");
    println!("  Iterations: {}", est.iterations);
    println!("  Sigma: {:.3} m", est.sigma_m);
    println!("  Residual RMS: {:.6} m\n", est.residual_rms);
}

fn bench_trilateration_2d() {
    let target: [f64; 2] = [40.0, 30.0];
    let anchors: Vec<[f64; 2]> = vec![[0.0, 0.0], [100.0, 0.0], [50.0, 86.6], [0.0, 100.0]];
    let ranges: Vec<f64> = anchors
        .iter()
        .map(|a| {
            let dx: f64 = a[0] - target[0];
            let dy: f64 = a[1] - target[1];
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    let sigmas = vec![1.0; 4];
    let trusts = vec![1.0; 4];

    let n = 10_000;
    let start = Instant::now();
    for _ in 0..n {
        let _ = trilaterate_2d(&anchors, &ranges, &sigmas, &trusts);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / n;

    let est = trilaterate_2d(&anchors, &ranges, &sigmas, &trusts).unwrap();
    let error =
        ((est.position[0] - target[0]).powi(2) + (est.position[1] - target[1]).powi(2)).sqrt();

    println!("2D Trilateration (4 anchors):");
    println!("  {n} iterations: {elapsed:?} ({per_call:?}/call)");
    println!("  Position error: {error:.6} m");
    println!("  Iterations: {}", est.iterations);
    println!("  Sigma: {:.3} m\n", est.sigma_m);
}

fn bench_ekf_convergence() {
    let true_pos: [f64; 3] = [50.0, 30.0, -20.0];
    let anchors: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 100.0],
    ];

    let config = FilterConfig {
        position_noise: 0.1,
        velocity_noise: 0.01,
        max_covariance: 1e8,
        innovation_gate_sigma: 4.0,
    };

    let n = 1_000;
    let start = Instant::now();
    for _ in 0..n {
        let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 200.0, config.clone());
        for _ in 0..20 {
            filter.predict(1.0);
            for anchor in &anchors {
                let dx: f64 = true_pos[0] - anchor[0];
                let dy: f64 = true_pos[1] - anchor[1];
                let dz: f64 = true_pos[2] - anchor[2];
                let range = (dx * dx + dy * dy + dz * dz).sqrt();
                filter.update(anchor, range, 5.0, 1.0);
            }
        }
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / n;

    // Final accuracy
    let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 200.0, config);
    for _ in 0..20 {
        filter.predict(1.0);
        for anchor in &anchors {
            let dx: f64 = true_pos[0] - anchor[0];
            let dy: f64 = true_pos[1] - anchor[1];
            let dz: f64 = true_pos[2] - anchor[2];
            let range = (dx * dx + dy * dy + dz * dz).sqrt();
            filter.update(anchor, range, 5.0, 1.0);
        }
    }
    let pos = filter.position();
    let error = ((pos[0] - true_pos[0]).powi(2)
        + (pos[1] - true_pos[1]).powi(2)
        + (pos[2] - true_pos[2]).powi(2))
    .sqrt();

    println!("EKF Convergence (20 rounds × 4 anchors = 80 updates):");
    println!("  {n} full convergence runs: {elapsed:?} ({per_call:?}/run)");
    println!("  Final position error: {error:.3} m");
    println!("  Final sigma: {:.3} m", filter.position_sigma());
    println!("  Update count: {}\n", filter.state.update_count);
}

fn bench_pdr_walkabout() {
    use positioning::dead_reckoning::*;

    // Simulate a 500-step walk around a square (125 steps per side)
    let steps_per_side = 125;
    let config = PdrConfig::default();

    let n = 1_000;
    let start = Instant::now();
    for _ in 0..n {
        let mut pdr = PedestrianDeadReckoning::new(config.clone());
        // Walk north
        pdr.set_heading(0.0);
        for _ in 0..steps_per_side {
            pdr.step(10.5, 0.0, 1013.0, 1, 0.5);
        }
        // Walk east
        pdr.set_heading(90.0);
        for _ in 0..steps_per_side {
            pdr.step(10.5, 0.0, 1013.0, 1, 0.5);
        }
        // Walk south
        pdr.set_heading(180.0);
        for _ in 0..steps_per_side {
            pdr.step(10.5, 0.0, 1013.0, 1, 0.5);
        }
        // Walk west
        pdr.set_heading(270.0);
        for _ in 0..steps_per_side {
            pdr.step(10.5, 0.0, 1013.0, 1, 0.5);
        }
    }
    let elapsed = start.elapsed();
    let per_run = elapsed / n;

    // Final accuracy check
    let mut pdr = PedestrianDeadReckoning::new(config);
    pdr.set_heading(0.0);
    for _ in 0..steps_per_side {
        pdr.step(10.5, 0.0, 1013.0, 1, 0.5);
    }
    pdr.set_heading(90.0);
    for _ in 0..steps_per_side {
        pdr.step(10.5, 0.0, 1013.0, 1, 0.5);
    }
    pdr.set_heading(180.0);
    for _ in 0..steps_per_side {
        pdr.step(10.5, 0.0, 1013.0, 1, 0.5);
    }
    pdr.set_heading(270.0);
    for _ in 0..steps_per_side {
        pdr.step(10.5, 0.0, 1013.0, 1, 0.5);
    }

    let pos = pdr.get_position();
    let error = (pos[0].powi(2) + pos[1].powi(2)).sqrt();
    let distance_walked = steps_per_side as f64 * 4.0 * pdr.stride_length();

    println!(
        "PDR Walkabout (500 steps, square, ~{:.0}m total):",
        distance_walked
    );
    println!(
        "  {n} full walks: {elapsed:?} ({per_run:?}/walk, {:.1}µs/step)",
        per_run.as_nanos() as f64 / 500.0 / 1000.0
    );
    println!("  Return-to-origin error: {error:.3}m");
    println!("  Drift estimate: {:.1}m", pdr.get_drift_estimate());
    println!("  Total steps: {}\n", pdr.total_steps());
}

fn bench_ranging_models() {
    use positioning::ranging::*;

    let n = 100_000;

    let start = Instant::now();
    for _ in 0..n {
        rssi_to_range(-65.0, -50.0, 2.5);
    }
    println!(
        "RSSI ranging:  {n} calls in {:?} ({:?}/call)",
        start.elapsed(),
        start.elapsed() / n
    );

    let start = Instant::now();
    for _ in 0..n {
        uwb_tof_to_range(33.36);
    }
    println!(
        "UWB ranging:   {n} calls in {:?} ({:?}/call)",
        start.elapsed(),
        start.elapsed() / n
    );

    let start = Instant::now();
    for _ in 0..n {
        lora_toa_to_range(3336.0, 125_000.0);
    }
    println!(
        "LoRa ranging:  {n} calls in {:?} ({:?}/call)",
        start.elapsed(),
        start.elapsed() / n
    );

    let start = Instant::now();
    for _ in 0..n {
        wifi_rtt_to_range(66.7);
    }
    println!(
        "WiFi RTT:      {n} calls in {:?} ({:?}/call)\n",
        start.elapsed(),
        start.elapsed() / n
    );
}

fn bench_gdop() {
    let anchors = vec![
        [100.0, 0.0, 0.0],
        [-50.0, 86.6, 0.0],
        [-50.0, -86.6, 0.0],
        [0.0, 0.0, 100.0],
    ];
    let point = [0.0, 0.0, 0.0];

    let n = 100_000;
    let start = Instant::now();
    for _ in 0..n {
        gdop(&anchors, &point);
    }
    let elapsed = start.elapsed();
    let g = gdop(&anchors, &point);

    println!("GDOP computation:");
    println!("  {n} calls: {elapsed:?} ({:?}/call)", elapsed / n);
    println!("  GDOP value: {g:.3} (< 3 = good)\n");
}

fn bench_geodetic_roundtrip() {
    let body = Earth;
    let n = 100_000;

    let start = Instant::now();
    for i in 0..n {
        let lat = -90.0 + (i as f64 / n as f64) * 180.0;
        let xyz = body.geodetic_to_cartesian(lat, 28.0, 1753.0);
        let _ = body.cartesian_to_geodetic(xyz);
    }
    let elapsed = start.elapsed();

    println!("Geodetic ↔ ECEF roundtrip (Earth WGS-84):");
    println!("  {n} roundtrips: {elapsed:?} ({:?}/call)\n", elapsed / n);
}
