// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Automated Earth model calibration via Nelder-Mead optimization.
//! Minimizes combined MAPE across population, emissions, and temperature
//! against observed 1970-2024 data.
//!
//! Free parameters:
//! 0. birth_rate_scale (multiplier on fertility_by_age base rates)
//! 1. edu_fertility_threshold (education index above which fertility drops)
//! 2. edu_fertility_strength (how much education reduces fertility)
//! 3. decarbonization_rate (annual carbon intensity decline)
//! 4. ocean_lag_years (thermal inertia e-folding time)

use mycelix_multiworld_sim::{
    earth_population::EarthPopulationModel,
    earth_regions::build_earth_regions,
    stochastic::StochasticEngine,
    validation::{self, ObservedData},
    viability::ScalingFactors,
};

/// Run the Earth model for 54 years and return combined MAPE.
fn evaluate(params: &[f64; 5]) -> f64 {
    let birth_scale = params[0];
    let edu_threshold = params[1];
    let edu_strength = params[2];
    let decarb_rate = params[3];
    let ocean_lag = params[4];

    // Bounds check — reject obviously bad parameters
    if birth_scale < 0.5 || birth_scale > 2.0 {
        return 1000.0;
    }
    if edu_threshold < 0.2 || edu_threshold > 0.8 {
        return 1000.0;
    }
    if edu_strength < 0.0 || edu_strength > 3.0 {
        return 1000.0;
    }
    if decarb_rate < 0.005 || decarb_rate > 0.04 {
        return 1000.0;
    }
    if ocean_lag < 5.0 || ocean_lag > 50.0 {
        return 1000.0;
    }

    // Build Earth model with 1970 initial conditions
    let mut regions = build_earth_regions();
    let scale_1970 = 3.70 / regions.iter().map(|r| r.population).sum::<f64>() * 1000.0;
    for region in &mut regions {
        region.population *= scale_1970;
        region.gdp_per_capita *= 0.33;
        region.education_index *= 0.7;
    }

    let mut model = EarthPopulationModel::from_regions(&regions);

    // Apply calibration parameters
    // birth_scale: modify initial fertility rates in all cohorts
    for demo in &mut model.demographics {
        for cohort in demo.cohorts.values_mut() {
            cohort.fertility_rate *= birth_scale;
        }
    }

    // Override climate parameters
    model.climate.carbon_intensity = 0.55; // 1970 starting value
                                           // We can't directly set decarb_rate and ocean_lag on the model struct,
                                           // so we'll track them externally and compute manually

    let mut regions_mut = regions.clone();
    let mut rng = StochasticEngine::new(42);

    let mut pop_trajectory = Vec::new();
    let mut temp_trajectory = Vec::new();
    let mut emission_trajectory = Vec::new();

    // Custom climate tracking (to apply calibrated params)
    let mut cumulative_gt = 200.0_f64; // baseline
    let mut temp_anomaly = 0.36_f64; // 1970 start
    let mut carbon_intensity = 0.55_f64;

    for tick in 0..648 {
        let scaling: Vec<_> = regions_mut
            .iter()
            .map(|r| ScalingFactors::compute(r.population * 1_000_000.0))
            .collect();
        model.tick(&regions_mut, &scaling, tick, &mut rng);

        // Apply calibrated GDP growth + education (reuse existing sync logic)
        model.sync_to_regions(&mut regions_mut);

        // Override education-fertility in demographics
        let years_elapsed = tick as f64 / 12.0;
        for demo in &mut model.demographics {
            let edu = demo.mean_education();
            let edu_penalty = (edu - edu_threshold).max(0.0) * edu_strength;
            let adjusted_tfr = (demo.total_fertility_rate - edu_penalty * 0.01).max(1.2);
            demo.total_fertility_rate = adjusted_tfr;
        }

        // Custom carbon intensity with calibrated decarb rate
        carbon_intensity = 0.55 * (1.0 - decarb_rate).powf(years_elapsed);

        // Emissions from GDP
        let global_gdp: f64 = regions_mut
            .iter()
            .map(|r| r.gdp_per_capita * r.population * 1_000_000.0)
            .sum();
        let annual_emissions = global_gdp * carbon_intensity / 1e12;

        // Cumulative + temperature with calibrated ocean lag
        cumulative_gt += annual_emissions / 12.0;
        let equilibrium = cumulative_gt * 1.8 / 1000.0;
        let target = equilibrium * 0.55;
        let lag_rate = 1.0 / (ocean_lag * 12.0);
        temp_anomaly += (target - temp_anomaly) * lag_rate;

        let year = 1970.0 + years_elapsed;
        if tick % 12 == 0 {
            pop_trajectory.push((year, model.total_population / 1000.0));
            temp_trajectory.push((year, temp_anomaly));
            emission_trajectory.push((year, annual_emissions));
        }
    }

    // Compute MAPEs
    let pop_v = validation::validate_metric(
        "pop",
        &pop_trajectory,
        &ObservedData::world_population(),
        "",
    );
    let temp_v = validation::validate_metric(
        "temp",
        &temp_trajectory,
        &ObservedData::temperature_anomaly(),
        "",
    );
    let emit_v = validation::validate_metric(
        "emit",
        &emission_trajectory,
        &ObservedData::co2_emissions(),
        "",
    );

    // Combined objective: weighted sum of MAPEs
    // Weight temperature less because early years have natural variability
    let combined = pop_v.mape * 1.0 + emit_v.mape * 1.0 + temp_v.mape * 0.3;
    combined
}

fn main() {
    println!("=== AUTOMATED EARTH MODEL CALIBRATION ===\n");
    println!("Optimizer: Nelder-Mead simplex");
    println!("Objective: minimize weighted MAPE (pop×1 + emit×1 + temp×0.3)\n");

    // Initial simplex: current best + perturbations
    let x0 = [1.0, 0.5, 1.0, 0.015, 20.0]; // current defaults
    let initial_cost = evaluate(&x0);
    println!(
        "Initial params: birth={:.2}, edu_thresh={:.2}, edu_str={:.2}, decarb={:.4}, lag={:.1}",
        x0[0], x0[1], x0[2], x0[3], x0[4]
    );
    println!("Initial cost: {:.2}\n", initial_cost);

    // Nelder-Mead simplex optimization
    let n = 5;
    let mut simplex: Vec<([f64; 5], f64)> = Vec::new();

    // Generate initial simplex: base point + n perturbations
    simplex.push((x0, initial_cost));
    let perturbations = [
        [1.15, 0.5, 1.0, 0.015, 20.0],
        [1.0, 0.45, 1.0, 0.015, 20.0],
        [1.0, 0.5, 0.7, 0.015, 20.0],
        [1.0, 0.5, 1.0, 0.018, 20.0],
        [1.0, 0.5, 1.0, 0.015, 25.0],
    ];
    for p in &perturbations {
        let cost = evaluate(p);
        simplex.push((*p, cost));
        eprint!(".");
    }
    eprintln!();

    // Nelder-Mead iterations
    let max_iter = 100;
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    for iter in 0..max_iter {
        // Sort by cost
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let best_cost = simplex[0].1;
        let worst_cost = simplex[n].1;

        if iter % 10 == 0 {
            println!(
                "Iter {:>3}: best={:.2} worst={:.2} params=[{:.3}, {:.3}, {:.3}, {:.4}, {:.1}]",
                iter,
                best_cost,
                worst_cost,
                simplex[0].0[0],
                simplex[0].0[1],
                simplex[0].0[2],
                simplex[0].0[3],
                simplex[0].0[4]
            );
        }

        // Convergence check
        if (worst_cost - best_cost).abs() < 0.5 {
            println!("Converged at iteration {}", iter);
            break;
        }

        // Centroid (excluding worst)
        let mut centroid = [0.0; 5];
        for i in 0..n {
            for j in 0..5 {
                centroid[j] += simplex[i].0[j];
            }
        }
        for j in 0..5 {
            centroid[j] /= n as f64;
        }

        // Reflection
        let mut reflected = [0.0; 5];
        for j in 0..5 {
            reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[n].0[j]);
        }
        let reflected_cost = evaluate(&reflected);

        if reflected_cost < simplex[n - 1].1 && reflected_cost >= simplex[0].1 {
            simplex[n] = (reflected, reflected_cost);
            continue;
        }

        if reflected_cost < simplex[0].1 {
            // Expansion
            let mut expanded = [0.0; 5];
            for j in 0..5 {
                expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
            }
            let expanded_cost = evaluate(&expanded);
            if expanded_cost < reflected_cost {
                simplex[n] = (expanded, expanded_cost);
            } else {
                simplex[n] = (reflected, reflected_cost);
            }
            continue;
        }

        // Contraction
        let mut contracted = [0.0; 5];
        for j in 0..5 {
            contracted[j] = centroid[j] + rho * (simplex[n].0[j] - centroid[j]);
        }
        let contracted_cost = evaluate(&contracted);
        if contracted_cost < simplex[n].1 {
            simplex[n] = (contracted, contracted_cost);
            continue;
        }

        // Shrink
        let best = simplex[0].0;
        for i in 1..=n {
            for j in 0..5 {
                simplex[i].0[j] = best[j] + sigma * (simplex[i].0[j] - best[j]);
            }
            simplex[i].1 = evaluate(&simplex[i].0);
        }
    }

    // Final result
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let best = &simplex[0];
    println!("\n=== CALIBRATION RESULT ===\n");
    println!("Optimized parameters:");
    println!("  birth_rate_scale:     {:.4}", best.0[0]);
    println!("  edu_fertility_thresh: {:.4}", best.0[1]);
    println!("  edu_fertility_str:    {:.4}", best.0[2]);
    println!("  decarbonization_rate: {:.5}", best.0[3]);
    println!("  ocean_lag_years:      {:.2}", best.0[4]);
    println!("  combined_cost:        {:.2} (lower = better)", best.1);

    // Run final validation with best params
    println!("\nFinal validation with optimized params:");
    let cost = evaluate(&best.0);
    println!("  Confirmed cost: {:.2}", cost);
}
