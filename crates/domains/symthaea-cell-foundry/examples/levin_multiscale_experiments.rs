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
    AdaptiveCouplingConfig, Perturbation, build_linear_axis_template,
    build_radial_bipolar_template, run_adaptive_coupling_experiment,
    run_counterfactual_coupling_experiment, run_dose_response_experiment,
};

const SEEDS: [u64; 4] = [17, 29, 41, 53];
const PERMEABILITIES: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const CELLS: usize = 128;
const MATURATION_DAYS: u32 = 12;
const RECOVERY_DAYS: u32 = 30;
const RECOVERY_TOLERANCE: f64 = 0.05;

#[derive(Debug, Serialize)]
struct AdaptiveRunRecord {
    controller: String,
    seed: u64,
    template_family: String,
    perturbation: String,
    homing_enabled: bool,
    initial_permeability: f32,
    final_applied_permeability: f32,
    final_discrepancy: f64,
    integrated_discrepancy: f64,
    fixed_control_mean_discrepancy: f64,
    adaptive_minus_fixed_mean: f64,
    first_day_below_tolerance: Option<u32>,
    wall_time_ms: u128,
    controller_trajectory: Vec<(u32, f32, f64)>,
}

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
    let mut adaptive_records = Vec::new();
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

                for homing_enabled in [false, true] {
                    let config = AdaptiveCouplingConfig::default();
                    for controller in ["trend", "counterfactual"] {
                        let start = Instant::now();
                        let adaptive = match controller {
                            "trend" => run_adaptive_coupling_experiment(
                                template, perturbation, RECOVERY_DAYS, homing_enabled, config,
                            )?,
                            "counterfactual" => run_counterfactual_coupling_experiment(
                                template, perturbation, RECOVERY_DAYS, homing_enabled, config,
                            )?,
                            _ => unreachable!("controller is from a fixed internal list"),
                        };
                        let wall_time_ms = start.elapsed().as_millis();
                        let control_values: Vec<f64> = records.iter()
                            .filter(|record| {
                                record.seed == seed
                                    && record.template_family == family
                                    && record.perturbation == adaptive.perturbation_label
                                    && record.homing_enabled == homing_enabled
                                    && [0.0_f32, 0.5_f32, 1.0_f32]
                                        .contains(&record.permeability)
                            })
                            .map(|record| record.final_discrepancy)
                            .collect();
                        if control_values.len() != 3 {
                            return Err(format!(
                                "expected 3 fixed controls for {family}/{}; got {}",
                                adaptive.perturbation_label, control_values.len()
                            ).into());
                        }
                        let fixed_control_mean_discrepancy =
                            control_values.iter().sum::<f64>() / control_values.len() as f64;
                        let final_applied_permeability = adaptive.permeability_trajectory
                            .last().map(|(_, value)| *value)
                            .unwrap_or(config.initial_permeability);
                        let controller_trajectory = adaptive.discrepancy_trajectory.iter()
                            .zip(&adaptive.permeability_trajectory)
                            .map(|((day, discrepancy), (_, permeability))| {
                                (*day, *permeability, *discrepancy)
                            })
                            .collect();
                        adaptive_records.push(AdaptiveRunRecord {
                            controller: controller.to_string(),
                            seed,
                            template_family: family.to_string(),
                            perturbation: adaptive.perturbation_label,
                            homing_enabled,
                            initial_permeability: config.initial_permeability,
                            final_applied_permeability,
                            final_discrepancy: adaptive.final_discrepancy,
                            integrated_discrepancy: adaptive.integrated_discrepancy,
                            fixed_control_mean_discrepancy,
                            adaptive_minus_fixed_mean: adaptive.final_discrepancy
                                - fixed_control_mean_discrepancy,
                            first_day_below_tolerance: adaptive.first_day_below_tolerance,
                            wall_time_ms,
                            controller_trajectory,
                        });
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

    println!();
    println!("ADAPTIVE COUPLING COMPARISON (paired per seed with fixed permeability 0.0/0.5/1.0 mean)");
    println!("controller,seed,template,perturbation,homing,initial_p,final_applied_p,final_error,integrated_error,fixed_mean_error,adaptive_minus_fixed_mean,recovery_day,wall_ms");
    let mut adaptive_grouped: BTreeMap<(String, String, bool, String), Vec<(f64, f64)>> = BTreeMap::new();
    for record in &adaptive_records {
        println!(
            "{},{},{},\"{}\",{},{:.2},{:.2},{:.6},{:.6},{:.6},{:.6},{},{}",
            record.controller,
            record.seed,
            record.template_family,
            record.perturbation,
            record.homing_enabled,
            record.initial_permeability,
            record.final_applied_permeability,
            record.final_discrepancy,
            record.integrated_discrepancy,
            record.fixed_control_mean_discrepancy,
            record.adaptive_minus_fixed_mean,
            record.first_day_below_tolerance.map_or_else(
                || "not_recovered".to_string(),
                |day| day.to_string()
            ),
            record.wall_time_ms
        );
        let summary_perturbation = if record.perturbation.starts_with("scramble_vmem(") {
            "scramble_vmem".to_string()
        } else {
            record.perturbation.clone()
        };
        adaptive_grouped.entry((
            record.template_family.clone(),
            summary_perturbation,
            record.homing_enabled,
            record.controller.clone(),
        )).or_default().push((
            record.adaptive_minus_fixed_mean,
            record.final_discrepancy
        ));
    }
    println!();
    println!("ADAPTIVE GROUP SUMMARY (four independent seeds; negative delta means adaptive beat fixed-control mean)");
    println!("template,perturbation,homing,controller,n,mean_adaptive_minus_fixed,sd_delta,mean_adaptive_error");
    for ((family, perturbation, homing, controller), values) in &adaptive_grouped {
        let deltas: Vec<f64> = values.iter().map(|(delta, _)| *delta).collect();
        let errors: Vec<f64> = values.iter().map(|(_, error)| *error).collect();
        let n = deltas.len() as f64;
        let mean_delta = deltas.iter().sum::<f64>() / n;
        let variance = deltas.iter().map(|delta| (delta - mean_delta).powi(2)).sum::<f64>() / n;
        let mean_error = errors.iter().sum::<f64>() / n;
        println!(
            "{},\"{}\",{},{},{},{:.6},{:.6},{:.6}",
            family, perturbation, homing, controller, values.len(), mean_delta, variance.sqrt(), mean_error
        );
    }
    let adaptive_json = serde_json::to_string_pretty(&adaptive_records)?;
    std::fs::write("levin_adaptive_coupling_results.json", &adaptive_json)?;
    eprintln!(
        "Wrote {} adaptive condition records to levin_adaptive_coupling_results.json ({} bytes).",
        adaptive_records.len(), adaptive_json.len()
    );
    Ok(())
}
