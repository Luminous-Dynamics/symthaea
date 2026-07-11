// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Property / invariant tests over the numeric domain crates — hardening them
//! past hand-picked ground truth by checking laws that must hold for *all*
//! valid inputs, sampled with a small deterministic PRNG.

use symthaea_economics::finance::npv;
use symthaea_economics::gini;
use symthaea_epidemiology::Sir;
use symthaea_epidemiology::sir::State;

/// Deterministic LCG → f64 in [0, 1).
fn rng(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
}

#[test]
fn gini_is_always_in_unit_interval() {
    let mut seed = 0x1234_5678u64;
    for _ in 0..2000 {
        let n = 1 + (rng(&mut seed) * 12.0) as usize;
        let values: Vec<f64> = (0..n).map(|_| rng(&mut seed) * 1000.0).collect();
        let g = gini(&values);
        assert!(
            (0.0..=1.0).contains(&g),
            "gini out of [0,1]: {g} for {values:?}"
        );
    }
}

#[test]
fn gini_of_identical_values_is_zero() {
    let mut seed = 0x9999u64;
    for _ in 0..500 {
        let v = rng(&mut seed) * 500.0 + 1.0;
        let n = 1 + (rng(&mut seed) * 10.0) as usize;
        let equal = vec![v; n];
        assert!(gini(&equal).abs() < 1e-9);
    }
}

#[test]
fn sir_conserves_population_for_all_params() {
    let mut seed = 0xABCDu64;
    for _ in 0..1000 {
        let beta = rng(&mut seed) * 0.9 + 0.01;
        let gamma = rng(&mut seed) * 0.4 + 0.01;
        let i0 = rng(&mut seed) * 0.1 + 0.001;
        let start = State {
            s: 1.0 - i0,
            i: i0,
            r: 0.0,
        };
        let (end, _) = Sir { beta, gamma }.simulate(start, 0.05, 2000);
        assert!(
            (end.s + end.i + end.r - 1.0).abs() < 1e-6,
            "population not conserved: {}",
            end.s + end.i + end.r
        );
    }
}

#[test]
fn sir_final_size_is_bounded_and_thresholded() {
    let mut seed = 0x5555u64;
    for _ in 0..1000 {
        let beta = rng(&mut seed) * 0.9 + 0.01;
        let gamma = rng(&mut seed) * 0.4 + 0.01;
        let sir = Sir { beta, gamma };
        let z = sir.final_size();
        assert!((0.0..=1.0).contains(&z), "final size out of [0,1]: {z}");
        // No epidemic below the R0=1 threshold; an epidemic above it.
        if sir.basic_reproduction_number() <= 1.0 {
            assert_eq!(z, 0.0);
        } else {
            assert!(z > 0.0);
        }
    }
}

#[test]
fn npv_decreases_monotonically_with_discount_rate() {
    // For a conventional project (negative now, positive later), a higher
    // discount rate can only lower the NPV.
    let mut seed = 0x7777u64;
    for _ in 0..500 {
        let flows: Vec<f64> = std::iter::once(-(rng(&mut seed) * 1000.0 + 100.0))
            .chain((0..5).map(|_| rng(&mut seed) * 400.0))
            .collect();
        let low = npv(0.02, &flows);
        let high = npv(0.20, &flows);
        assert!(high <= low + 1e-9, "npv not monotone: {high} > {low}");
    }
}
