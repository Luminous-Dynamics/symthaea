// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Run the Earth model and compare against historical data (1970-2024).
//! This is the credibility test: can we reproduce the last 50 years?

use mycelix_multiworld_sim::{
    config::SimulationConfig,
    earth_population::EarthPopulationModel,
    earth_regions::build_earth_regions,
    validation::{self, format_validation_report, ObservedData},
    MultiWorldSimulator,
};

fn main() {
    println!("=== HISTORICAL VALIDATION: Earth Model vs Reality ===\n");

    // Build Earth regions and scale to 1970 population levels.
    // build_earth_regions() returns 2024 data (~7.8B total).
    // 1970 world population was 3.7B → scale factor = 3.7/7.8 ≈ 0.474.
    let mut regions = build_earth_regions();
    let scale_1970 = 3.70 / regions.iter().map(|r| r.population).sum::<f64>() * 1000.0;
    for region in &mut regions {
        region.population *= scale_1970;
        // 1970 GDP per capita was roughly 1/3 of 2024 levels
        region.gdp_per_capita *= 0.33;
        // Education was lower
        region.education_index *= 0.7;
    }
    let mut model = EarthPopulationModel::from_regions(&regions);
    let mut regions_mut = regions.clone();

    let initial_pop = model.total_population;
    println!(
        "Initial Earth population: {:.0}M ({:.2}B)",
        initial_pop,
        initial_pop / 1000.0
    );
    println!(
        "Initial global temp anomaly: {:.2}°C",
        model.climate.global_temp_anomaly
    );
    println!("Running 54 years (1970-2024) at monthly resolution...\n");

    // Track trajectories
    let mut pop_trajectory = Vec::new();
    let mut temp_trajectory = Vec::new();
    let mut emission_trajectory = Vec::new();
    let mut tfr_trajectory = Vec::new();

    let mut rng = mycelix_multiworld_sim::stochastic::StochasticEngine::new(42);
    let mut primitives =
        mycelix_multiworld_sim::primitives::CivilizationalPrimitives::earth_defaults();

    // Run 54 years (648 ticks)
    for tick in 0..648 {
        let scaling: Vec<_> = regions_mut
            .iter()
            .map(|r| {
                mycelix_multiworld_sim::viability::ScalingFactors::compute(
                    r.population * 1_000_000.0,
                )
            })
            .collect();
        model.tick(&regions_mut, &scaling, tick, &mut rng);
        model.sync_to_regions(&mut regions_mut);

        // Tick primitives with Earth aggregate state
        let total_pop: f64 = regions_mut.iter().map(|r| r.population).sum();
        let mean_urban: f64 = regions_mut
            .iter()
            .map(|r| r.urbanization * r.population)
            .sum::<f64>()
            / total_pop.max(1.0);
        let mean_gdp: f64 = regions_mut
            .iter()
            .map(|r| r.gdp_per_capita * r.population)
            .sum::<f64>()
            / total_pop.max(1.0);
        primitives.tick(total_pop, mean_urban, mean_gdp, 0.05, 0.3, 0.1);

        // Feedback: resource EROI → GDP (only when significantly depleted)
        if let Some(oil) = primitives.resources.iter().find(|r| r.name == "Oil") {
            let eroi = oil.current_eroi();
            if eroi < 8.0 {
                // GDP drag from low EROI: 0.5% per point below 8, monthly
                let monthly_drag = 1.0 - (8.0 - eroi) * 0.0004;
                for region in &mut regions_mut {
                    region.gdp_per_capita *= monthly_drag.max(0.998);
                }
            }
        }
        // Ecosystem agriculture modifier applied to GDP, not specialization
        // (specialization should not compound each tick)
        let ag_mod = primitives.ecosystem.agriculture_modifier();
        if ag_mod < 0.95 {
            for region in &mut regions_mut {
                region.gdp_per_capita *= 1.0 - (1.0 - ag_mod) * 0.01; // mild monthly drag
            }
        }

        let year = 1970.0 + tick as f64 / 12.0;

        // Record yearly snapshots
        if tick % 12 == 0 {
            pop_trajectory.push((year, model.total_population / 1000.0)); // billions
            temp_trajectory.push((year, model.climate.global_temp_anomaly));
            emission_trajectory.push((year, model.climate.annual_emissions));

            // Estimate global TFR from demographics
            let total_births: f64 = model
                .demographics
                .iter()
                .flat_map(|d| d.cohorts.values())
                .map(|c| c.fertility_rate * c.count / 12.0)
                .sum();
            let women_reproductive: f64 = model
                .demographics
                .iter()
                .flat_map(|d| d.cohorts.iter())
                .filter(|(k, _)| {
                    k.sex == mycelix_multiworld_sim::earth_population::cohort::CohortSex::Female
                        && k.age_band.can_reproduce()
                })
                .map(|(_, c)| c.count)
                .sum();
            let tfr_estimate = if women_reproductive > 0.0 {
                total_births / women_reproductive * 12.0 * 35.0 // births/woman × years of fertility
            } else {
                0.0
            };
            tfr_trajectory.push((year, tfr_estimate.clamp(0.5, 8.0)));
        }
    }

    // Validate against observed data
    let pop_validation = validation::validate_metric(
        "World Population (billions)",
        &pop_trajectory,
        &ObservedData::world_population(),
        "UN World Population Prospects 2024",
    );

    let temp_validation = validation::validate_metric(
        "Global Temperature Anomaly (°C)",
        &temp_trajectory,
        &ObservedData::temperature_anomaly(),
        "NASA GISS GISTEMP v4",
    );

    let emission_validation = validation::validate_metric(
        "CO₂ Emissions (GtCO₂/year)",
        &emission_trajectory,
        &ObservedData::co2_emissions(),
        "Global Carbon Project 2024",
    );

    let validations = vec![pop_validation, temp_validation, emission_validation];

    // Print results
    println!("{}", format_validation_report(&validations));

    // Summary
    let overall_mape: f64 =
        validations.iter().map(|v| v.mape).sum::<f64>() / validations.len() as f64;
    println!("=== VERDICT ===");
    if overall_mape < 15.0 {
        println!(
            "Overall MAPE: {:.1}% — Model is CREDIBLE for trend analysis",
            overall_mape
        );
    } else if overall_mape < 30.0 {
        println!(
            "Overall MAPE: {:.1}% — Model captures DIRECTION but not magnitude",
            overall_mape
        );
    } else {
        println!(
            "Overall MAPE: {:.1}% — Model needs RECALIBRATION",
            overall_mape
        );
    }

    println!("\nFinal state (year 2024):");
    println!(
        "  Population: {:.2}B (observed: 8.12B)",
        model.total_population / 1000.0
    );
    println!(
        "  Temp anomaly: {:.2}°C (observed: 1.29°C)",
        model.climate.global_temp_anomaly
    );
    println!(
        "  CO₂ emissions: {:.1} GtCO₂/yr (observed: 37.4)",
        model.climate.annual_emissions
    );

    // Primitives state at 2024
    println!("\n=== CIVILIZATIONAL PRIMITIVES (2024) ===");
    println!(
        "  Ecosystem service index: {:.3} (1.0=pristine, <0.3=critical)",
        primitives.ecosystem.service_index()
    );
    println!(
        "    Biodiversity: {:.3}  Forest: {:.3}  Soil: {:.3}  Ocean: {:.3}",
        primitives.ecosystem.biodiversity,
        primitives.ecosystem.forest_cover,
        primitives.ecosystem.soil_health,
        primitives.ecosystem.ocean_health
    );
    if let Some(oil) = primitives.resources.iter().find(|r| r.name == "Oil") {
        println!(
            "  Oil: {:.0}% remaining, EROI {:.1}:1",
            oil.fraction_remaining() * 100.0,
            oil.current_eroi()
        );
    }
    println!(
        "  Network: mean degree {:.0}, clustering {:.3}, path length {:.1}",
        primitives.network.mean_degree,
        primitives.network.clustering,
        primitives.network.avg_path_length
    );
    println!(
        "  Knowledge: {} known, {} adjacent possible, {} paradigm shifts",
        primitives.knowledge.known_count,
        primitives.knowledge.adjacent_possible,
        primitives.knowledge.paradigm_shifts
    );
    println!(
        "  Institutional lock-in: {:.3}, quality: {:.3}",
        primitives.institutions.lock_in_strength, primitives.institutions.quality
    );
    println!(
        "  Trust: {:.3}, betrayals: {}",
        primitives.trust.level, primitives.trust.betrayal_count
    );
    let tips = primitives.ecosystem.tipping_points();
    if !tips.is_empty() {
        println!("  TIPPING POINTS CROSSED:");
        for tip in &tips {
            println!("    - {}", tip);
        }
    }
}
