// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Factorial, seed-replicated synthetic tests inspired by Levin's work.
//!
//! This is a simulation benchmark, not a claim of biological validation.
//! It compares pattern family, perturbation, gap-junction permeability,
//! and positional-homing memory while recording recovery error and runtime.
//!
//! Run: cargo run -p symthaea-cell-foundry --example levin_multiscale_experiments

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use serde::Serialize;
use symthaea_cell_foundry::experiments::{
    Perturbation, build_linear_axis_template, build_radial_bipolar_template,
    run_dose_response_experiment,
};

const SEEDS: [u64; 4] = [17, 29, 41, 53];
const PERMEABILITIES: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const CELLS: usize = 128;
const MATURATION_DAYS: u32 = 12;
const RECOVERY_DAYS: u32 = 30;
const RECOVERY_TOLERANCE: f64 = 0.05;

#[derive(Debug, Serialize)]
struct RunRecord {
    seed: u64,
    template_family: String,
    perturbation: String,
    homing_enabled: bool,
    permeability: f32,
    final_discrepancy: f64,
    mean_trajectory_discrepancy: f64,
    first_day_below_tolerance: Option<u32>,
    wall_time_ms: u128,
    trajectory: Vec<(u32, f64)>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut records = Vec::new();
    let mut grouped_final_errors: BTreeMap<(String, String, bool, u16), Vec<f64>> =
        BTreeMap::new();

    for seed in SEEDS {
        let radial =
            build_radial_bipolar_template(seed, CELLS, MATURATION_DAYS, 0.2);
        let axis = build_linear_axis_template(seed, CELLS, MATURATION_DAYS);

        for (family, template) in [("radial", &radial), ("axis", &axis)] {
            let perturbations = if family == "radial" {
                vec![
                    Perturbation::Amputate { min_r: 1.1, max_r: 2.0 },
                    Perturbation::Amputate { min_r: 0.8, max_r: 2.0 },
                    Perturbation::ScrambleVmem { seed: seed ^ 0x5343_5241_4d42_4c45 },
                ]
            } else {
                vec![
                    Perturbation::AmputateAxis { min_x: 0.35, max_x: 2.0 },
                    Perturbation::AmputateAxis { min_x: 0.0, max_x: 2.0 },
                    Perturbation::ScrambleVmem { seed: seed ^ 0x5343_5241_4d42_4c45 },
                ]
            };

            for perturbation in &perturbations {
                for homing_enabled in [false, true] {
                    for permeability in PERMEABILITIES {
                        let start = Instant::now();
                        let condition = run_dose_response_experiment(
                            template,
                            perturbation,
                            RECOVERY_DAYS,
                            homing_enabled,
                            &[permeability],
                        )
                        .into_iter()
                        .next()
                        .expect("one permeability must produce one condition");
                        let wall_time_ms = start.elapsed().as_millis();
                        let mean_trajectory_discrepancy =
                            if condition.discrepancy_trajectory.is_empty() {
                                0.0
                            } else {
                                condition.discrepancy_trajectory.iter()
                                    .map(|(_, value)| value)
                                    .sum::<f64>()
                                    / condition.discrepancy_trajectory.len() as f64
                            };
                        let first_day_below_tolerance = condition.discrepancy_trajectory.iter()
                            .find(|entry| entry.1 <= RECOVERY_TOLERANCE)
                            .map(|entry| entry.0);
                        let label = condition.perturbation_label.clone();

                        records.push(RunRecord {
                            seed,
                            template_family: family.to_string(),
                            perturbation: label.clone(),
                            homing_enabled,
                            permeability,
                            final_discrepancy: condition.final_discrepancy,
                            mean_trajectory_discrepancy,
                            first_day_below_tolerance,
                            wall_time_ms,
                            trajectory: condition.discrepancy_trajectory.clone(),
                        });
                        let permeability_milli = (permeability * 1000.0).round() as u16;
                        let summary_perturbation = if label.starts_with("scramble_vmem(") {
                            "scramble_vmem".to_string()
                        } else {
                            label.clone()
                        };
                        grouped_final_errors
                            .entry((
                                family.to_string(),
                                summary_perturbation,
                                homing_enabled,
                                permeability_milli,
                            ))
                            .or_default()
                            .push(condition.final_discrepancy);
                    }
                }
            }
        }
    }

    println!("seed,template,perturbation,homing,permeability,final_error,mean_error,recovery_day,wall_ms");
    for record in &records {
        println!(
            "{},{},\"{}\",{},{:.2},{:.6},{:.6},{},{}",
            record.seed,
            record.template_family,
            record.perturbation,
            record.homing_enabled,
            record.permeability,
            record.final_discrepancy,
            record.mean_trajectory_discrepancy,
            record.first_day_below_tolerance.map_or_else(
                || "not_recovered".to_string(),
                |day| day.to_string()
            ),
            record.wall_time_ms,
        );
    }

    println!();
    println!("GROUP SUMMARY (four independent seeds; population standard deviation)");
    println!("template,perturbation,homing,permeability,n,mean_final_error,sd_final_error");
    for ((family, perturbation, homing, permeability_milli), values) in grouped_final_errors {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / n;
        println!(
            "{},\"{}\",{},{:.2},{},{:.6},{:.6}",
            family, perturbation, homing, permeability_milli as f64 / 1000.0,
            values.len(), mean, variance.sqrt()
        );
    }

    let json = serde_json::to_string_pretty(&records)?;
    std::fs::write("levin_multiscale_results.json", &json)?;
    eprintln!(
        "Wrote {} condition records to levin_multiscale_results.json ({} bytes).",
        records.len(), json.len()
    );
    Ok(())
}
