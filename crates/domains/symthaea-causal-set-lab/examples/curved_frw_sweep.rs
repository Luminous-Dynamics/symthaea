// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_causal_set_lab::ConformalFrw1p1;

fn halton(mut index: usize, base: usize) -> f64 {
    let mut factor = 1.0;
    let mut value = 0.0;
    while index > 0 {
        factor /= base as f64;
        value += factor * (index % base) as f64;
        index /= base;
    }
    value
}

/// Independent deterministic importance-quadrature reference for the
/// probability that two events are causally related in the curved measure.
/// The reference samples uniform null-coordinate points with Halton bases 2/3
/// and weights them by sqrt(|g|); it does not call the causal-set sampler or
/// the crate's related-pair counter.
fn qmc_ordering_fraction(kappa: f64, extent: f64, points: usize) -> f64 {
    let mut coordinates = Vec::with_capacity(points);
    let mut weights = Vec::with_capacity(points);
    let mut weight_sum = 0.0;
    let mut squared_weight_sum = 0.0;

    for index in 1..=points {
        let u = extent * halton(index, 2);
        let v = extent * halton(index, 3);
        let eta = 0.5 * (u + v) - 0.5 * extent;
        let weight = (kappa * eta * eta).exp();
        coordinates.push((u, v));
        weights.push(weight);
        weight_sum += weight;
        squared_weight_sum += weight * weight;
    }

    let mut related_weight = 0.0;
    for first in 0..points {
        let (u_first, v_first) = coordinates[first];
        for second in (first + 1)..points {
            let (u_second, v_second) = coordinates[second];
            let related = (u_first < u_second && v_first < v_second)
                || (u_second < u_first && v_second < v_first);
            if related {
                related_weight += weights[first] * weights[second];
            }
        }
    }

    2.0 * related_weight / (weight_sum * weight_sum - squared_weight_sum)
}

fn main() -> Result<(), Box<dyn Error>> {
    const EXTENT: f64 = 2.0;
    const EVENT_COUNT: usize = 4_000;
    const REFERENCE_POINTS: usize = 2_048;
    const SEED: u64 = 0xF12_2026;

    println!("Symthaea conformal FRW causal-set sampling sweep");
    println!("epistemic_status=CURVED_MEASURE_SAMPLER_AND_FINITE_SAMPLE_DIAGNOSTIC_NOT_CONTINUUM_LIMIT");
    println!("metric=ds2=a(eta)^2*(-deta^2+dx^2)");
    println!("scale_factor=exp(kappa*eta^2/2)");
    println!("ricci_convention=R=2*a''/a^3");
    println!("diamond=|eta|<=T/2, |x|<=T/2-|eta|");
    println!("extent={EXTENT}");
    println!("events={EVENT_COUNT}");
    println!("seed={SEED}");
    println!("qmc_reference_points={REFERENCE_POINTS}");
    println!("kappa,total_volume,expected_eta2,empirical_eta2,empirical_order_fraction,qmc_order_fraction,longest_chain,events_per_volume");

    for kappa in [0.0_f64, 0.2, 0.5, 1.0, 2.0] {
        let metric = ConformalFrw1p1::new(kappa, EXTENT)?;
        let causet = metric.sprinkle_fixed_cardinality(SEED, EVENT_COUNT)?;

        let empirical_eta2 = causet
            .events()
            .iter()
            .map(|event| {
                let eta = event.coordinate_time() - 0.5 * EXTENT;
                eta * eta
            })
            .sum::<f64>()
            / EVENT_COUNT as f64;

        let empirical_order_fraction = causet.ordering_fraction().unwrap_or(0.0);
        let qmc_reference = qmc_ordering_fraction(kappa, EXTENT, REFERENCE_POINTS);
        let longest_chain = causet.longest_chain_ids().len();
        let density = EVENT_COUNT as f64 / metric.total_volume();

        println!(
            "{kappa:.6},{:.12},{:.12},{empirical_eta2:.12},{empirical_order_fraction:.12},{qmc_reference:.12},{longest_chain},{density:.9}",
            metric.total_volume(),
            metric.expected_eta_squared(),
        );
    }

    println!("The kappa=0 row is the flat-space control; curved ordering fractions must not be inverted using the flat Myrheim-Meyer law.");
    println!("The QMC ordering reference is deterministic finite-sample quadrature, not an exact integral or a continuum-limit proof.");
    Ok(())
}
