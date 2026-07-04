// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced Bioelectric Experiments
//!
//! Six follow-on experiments beyond the core equifinality/homing results,
//! all built on the same `experiments.rs` harness:
//!
//! 1. Statistical robustness — does "open beats blocked" hold across many
//!    seeds, or was the earlier single-seed result lucky?
//! 2. Dose-response — recovery quality as a smooth function of
//!    gap-junction permeability, not just open/blocked.
//! 3. Phase diagram — homing rate x diffusion rate, characterizing when
//!    each recovery channel dominates.
//! 4. Cancer-as-defection — does a locally gap-junction-isolated,
//!    hyperproliferative region actually grow unchecked?
//! 5. Axis polarity — does blocking gap junctions during posterior
//!    regeneration default toward head-like identity (the model's analog
//!    of gap-junction-blockade-induced double-headed regenerants)?
//! 6. Free self-organization — what happens with no imposed target at all?
//!
//! Run: cargo run -p symthaea-cell-foundry --example advanced_experiments

use serde::Serialize;
use symthaea_cell_foundry::experiments::{
    Perturbation, build_linear_axis_template, build_radial_bipolar_template, mean_vmem_in_x_band,
    run_dose_response_experiment, run_equifinality_experiment,
};
use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

#[derive(Serialize)]
struct Report {
    statistical_robustness: StatisticalRobustness,
    dose_response: Vec<(f32, f64)>,
    phase_diagram: Vec<PhasePoint>,
    defection_growth: Vec<(u32, usize)>,
    axis_polarity: AxisPolarity,
    free_self_organization: FreeSelfOrganization,
}

#[derive(Serialize)]
struct StatisticalRobustness {
    n_seeds: usize,
    open_wins: usize,
    mean_open: f64,
    mean_blocked: f64,
}

#[derive(Serialize)]
struct PhasePoint {
    homing_rate: f32,
    diffusion_rate: f32,
    final_discrepancy: f64,
}

#[derive(Serialize)]
struct AxisPolarity {
    open_band_vmem: Option<f32>,
    blocked_band_vmem: Option<f32>,
}

#[derive(Serialize)]
struct FreeSelfOrganization {
    days: u32,
    final_cells: usize,
    vmem_mean: f32,
    vmem_std: f32,
    neuron_fraction: f64,
}

fn main() {
    println!("======================================================");
    println!("   Advanced Bioelectric Experiments");
    println!("======================================================\n");

    // ---- 1. Statistical robustness ----
    println!("=== 1. Statistical robustness (multi-seed) ===");
    let seeds: Vec<u64> = vec![11, 22, 33, 44, 55];
    let mut opens = Vec::new();
    let mut blockeds = Vec::new();
    let mut open_wins = 0;
    for &seed in &seeds {
        let template = build_radial_bipolar_template(seed, 200, 20, 0.2);
        let perturbations = [
            Perturbation::Amputate {
                min_r: 0.9,
                max_r: 2.0,
            },
            Perturbation::Amputate {
                min_r: 0.8,
                max_r: 2.0,
            },
            Perturbation::Amputate {
                min_r: 0.7,
                max_r: 2.0,
            },
        ];
        let result = run_equifinality_experiment(&template, &perturbations, 40, false);
        let (open_mean, blocked_mean) = result.mean_final_by_permeability();
        opens.push(open_mean);
        blockeds.push(blocked_mean);
        if result.open_beats_blocked() {
            open_wins += 1;
        }
        println!(
            "  seed={seed}: open={open_mean:.4} blocked={blocked_mean:.4} open_wins={}",
            result.open_beats_blocked()
        );
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let robustness = StatisticalRobustness {
        n_seeds: seeds.len(),
        open_wins,
        mean_open: mean(&opens),
        mean_blocked: mean(&blockeds),
    };
    println!(
        "  -> open beat blocked in {}/{} seeds (mean open={:.4}, mean blocked={:.4})\n",
        robustness.open_wins, robustness.n_seeds, robustness.mean_open, robustness.mean_blocked
    );

    // ---- 2. Dose-response ----
    println!("=== 2. Dose-response (permeability sweep) ===");
    let dr_template = build_radial_bipolar_template(11, 200, 20, 0.2);
    let dr_perturbation = Perturbation::Amputate {
        min_r: 0.8,
        max_r: 2.0,
    };
    let permeabilities = [0.0f32, 0.25, 0.5, 0.75, 1.0];
    let dr_results =
        run_dose_response_experiment(&dr_template, &dr_perturbation, 40, false, &permeabilities);
    let dose_response: Vec<(f32, f64)> = dr_results
        .iter()
        .map(|c| (c.gap_junction_permeability, c.final_discrepancy))
        .collect();
    for (p, d) in &dose_response {
        println!("  permeability={p:.2} -> final discrepancy={d:.4}");
    }
    println!();

    // ---- 3. Phase diagram: homing rate x diffusion rate ----
    // Gap junctions OPEN here (permeability=1.0) so both channels are live
    // and can trade off against each other -- with them blocked, the
    // diffusion term is multiplied by permeability=0 and trivially vanishes
    // regardless of rate, which would make this diagram uninformative.
    println!("=== 3. Phase diagram (homing rate x diffusion rate, open GJ) ===");
    let pd_template = build_radial_bipolar_template(21, 200, 20, 0.2);
    let homing_rates = [0.0f32, 0.05, 0.15, 0.30];
    let diffusion_rates = [0.05f32, 0.15, 0.35, 0.60];
    let mut phase_diagram = Vec::new();
    println!(
        "  homing\\diffusion  {:>7} {:>7} {:>7} {:>7}",
        0.05, 0.15, 0.35, 0.60
    );
    for &h in &homing_rates {
        let mut row = format!("  {h:>15.2}  ");
        for &d in &diffusion_rates {
            let mut o = pd_template.clone();
            o.set_gap_junction_permeability(1.0); // open -- both channels live
            o.set_positional_homing(h > 0.0);
            o.set_positional_homing_rate(h);
            o.set_gap_junction_diffusion_rate(d);
            o.scramble_vmem(321);
            for _ in 0..40 {
                o.advance_day();
            }
            let disc = o.morphology_discrepancy().unwrap_or(0.0);
            phase_diagram.push(PhasePoint {
                homing_rate: h,
                diffusion_rate: d,
                final_discrepancy: disc,
            });
            row.push_str(&format!("{disc:>7.3} "));
        }
        println!("{row}");
    }
    println!("  (both channels live: recovery should improve with either rate rising.)\n");

    // ---- 4. Cancer-as-defection ----
    println!("=== 4. Cancer-as-defection (unregulated growth) ===");
    let mut cancer = build_radial_bipolar_template(41, 200, 20, 0.2);
    // r < 0.15 was empirically too small a region to reliably contain any
    // cells at this density (~229 cells in a [-1,1]^3 cube -> well under 1
    // expected cell in a sphere that size); 0.5 gives a robust handful.
    let removed_or_marked = cancer.induce_local_defection(|p| {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        r < 0.5
    });
    println!("  marked {removed_or_marked} cells as defected at t=0");
    let mut defection_growth = vec![(0u32, cancer.defected_cell_count())];
    // Unregulated growth compounds fast (both the defected population at
    // DEFECTION_PROLIFERATION_RATE=0.35/day and the rest of the tissue at
    // the baseline 0.10/day). This used to be capped at 3,000 cells / 14
    // days because `neighbours()` was an O(n) brute-force scan called
    // O(n) times per day -- O(n^2) daily cost that made a demo run
    // impractically slow past that point. `neighbours()` is now backed
    // by `crate::spatial_grid` (O(1)-amortized per query), so the same
    // window now runs a much larger population comfortably; the cap
    // below is a generous safety net, not a workaround for a performance
    // wall.
    let cancer_start = std::time::Instant::now();
    for day in 1..=30 {
        cancer.advance_day();
        if day % 2 == 0 {
            defection_growth.push((day, cancer.defected_cell_count()));
        }
        if cancer.field.num_cells() > 8000 {
            println!(
                "  (stopping early at day {day}: {} total cells)",
                cancer.field.num_cells()
            );
            break;
        }
    }
    println!(
        "  ({} days, {} final cells, {:.2}s wall-clock)",
        defection_growth.last().map(|(d, _)| *d).unwrap_or(0),
        cancer.field.num_cells(),
        cancer_start.elapsed().as_secs_f64()
    );
    for (day, count) in &defection_growth {
        println!(
            "  day {day:3}: {count} defected cells (total {})",
            cancer.field.num_cells()
        );
    }
    println!();

    // ---- 5. Axis polarity ----
    println!("=== 5. Axis polarity (double-headed-worm analog) ===");
    let axis_template = build_linear_axis_template(31, 200, 20);
    let cut_x = 0.4;
    let mut axis_open = axis_template.clone();
    axis_open.set_gap_junction_permeability(1.0);
    axis_open.amputate_where(|p| p[0] >= cut_x && p[0] < 2.0);
    let mut axis_blocked = axis_template.clone();
    axis_blocked.set_gap_junction_permeability(0.0);
    axis_blocked.amputate_where(|p| p[0] >= cut_x && p[0] < 2.0);
    for _ in 0..40 {
        axis_open.advance_day();
        axis_blocked.advance_day();
    }
    let open_band = mean_vmem_in_x_band(&axis_open, cut_x - 0.2, cut_x);
    let blocked_band = mean_vmem_in_x_band(&axis_blocked, cut_x - 0.2, cut_x);
    println!(
        "  cut-face region mean Vmem: open={:?}, blocked={:?} (head={:.1}, tail={:.1})",
        open_band,
        blocked_band,
        symthaea_cell_foundry::experiments::AXIS_HEAD_VMEM,
        symthaea_cell_foundry::experiments::AXIS_TAIL_VMEM
    );
    println!();

    // ---- 6. Free self-organization ----
    println!("=== 6. Free self-organization (no imposed target) ===");
    let mut free = NeuralOrganoid::new(150, 51);
    for _ in 0..40 {
        free.advance_day();
    }
    let vmems: Vec<f32> = (0..free.field.num_cells())
        .map(|i| free.field.bioelectric.vmem[i])
        .collect();
    let vmem_mean = vmems.iter().sum::<f32>() / vmems.len() as f32;
    let vmem_var = vmems.iter().map(|v| (v - vmem_mean).powi(2)).sum::<f32>() / vmems.len() as f32;
    let free_result = FreeSelfOrganization {
        days: 40,
        final_cells: free.field.num_cells(),
        vmem_mean,
        vmem_std: vmem_var.sqrt(),
        neuron_fraction: free.neuron_fraction(),
    };
    println!(
        "  {} days, {} cells, Vmem mean={:.3} std={:.3}, neuron fraction={:.4}",
        free_result.days,
        free_result.final_cells,
        free_result.vmem_mean,
        free_result.vmem_std,
        free_result.neuron_fraction
    );
    println!(
        "  (with no captured target, there's nothing to recover toward -- this is \
         just characterizing the model's unguided baseline.)\n"
    );

    let report = Report {
        statistical_robustness: robustness,
        dose_response,
        phase_diagram,
        defection_growth,
        axis_polarity: AxisPolarity {
            open_band_vmem: open_band,
            blocked_band_vmem: blocked_band,
        },
        free_self_organization: free_result,
    };
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write("advanced_experiments.json", &json).expect("write advanced_experiments.json");
    println!("Wrote advanced_experiments.json ({} bytes)", json.len());
}
