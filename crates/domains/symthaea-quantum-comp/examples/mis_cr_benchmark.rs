//! Compare the adaptive MIS planner with uniform sampling, Successive Rejects,
//! and both Continuous Rejects variants on paired synthetic Bernoulli streams.
//!
//! Usage:
//! cargo run -p symthaea-quantum-comp --example mis_cr_benchmark -- 2000 20261022 > trials.csv 2> summary.txt
//!
//! This is a statistical policy diagnostic, not an MIS solve or physical-QPU model.

use std::env;
use std::error::Error;

use symthaea_quantum_comp::mis_continuous_rejects::{
    MisContinuousRejectsVariant, run_mis_continuous_rejects,
};
use symthaea_quantum_comp::mis_shot_budget::MisShotBudgetConfig;
use symthaea_quantum_comp::mis_shot_simulation::{
    MisSamplingScenario, aggregate_mis_sampling_trials, simulate_mis_sampling_scenario,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let trials: usize = args.next().unwrap_or_else(|| "2000".to_owned()).parse()?;
    let master_seed: u64 = args.next().unwrap_or_else(|| "20261022".to_owned()).parse()?;
    if args.next().is_some() {
        return Err("usage: mis_cr_benchmark [trials] [master_seed]".into());
    }
    let config = MisShotBudgetConfig::new(8, 8, 512, 0.05)?;
    let scenarios = vec![
        MisSamplingScenario::new("wide-four", vec![0.80, 0.45, 0.20, 0.10])?,
        MisSamplingScenario::new("moderate-four", vec![0.60, 0.55, 0.40, 0.35])?,
        MisSamplingScenario::new("narrow-four", vec![0.55, 0.53, 0.51, 0.45])?,
        MisSamplingScenario::new("exact-tie-four", vec![0.50, 0.50, 0.50, 0.50])?,
    ];

    println!("record_type,scenario,trial_index,trial_seed,probabilities,true_best_vertex,adaptive_selected,adaptive_correct,adaptive_observations,uniform_selected,uniform_correct,uniform_observations,successive_rejects_selected,successive_rejects_correct,successive_rejects_observations,cr_c_selected,cr_c_correct,cr_c_observations,cr_c_eliminated,cr_a_selected,cr_a_correct,cr_a_observations,cr_a_eliminated");
    eprintln!("config,trials={trials},master_seed={master_seed},budget={},initial_shots_per_candidate={},batch_shots_per_candidate={},alpha={}", config.max_total_shots, config.initial_shots_per_candidate, config.batch_shots_per_candidate, config.familywise_error_probability);

    for scenario in scenarios {
        let rows = simulate_mis_sampling_scenario(&scenario, config, trials, master_seed)?;
        let best_mean = scenario.inclusion_probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut cr_c_correct = 0_u64;
        let mut cr_a_correct = 0_u64;
        let mut cr_c_total = 0_u128;
        let mut cr_a_total = 0_u128;
        let mut cr_c_eliminated = 0_u64;
        let mut cr_a_eliminated = 0_u64;

        for row in &rows {
            let conservative = run_mis_continuous_rejects(
                &scenario.inclusion_probabilities,
                config.max_total_shots,
                row.trial_seed,
                MisContinuousRejectsVariant::Conservative,
            )?;
            let aggressive = run_mis_continuous_rejects(
                &scenario.inclusion_probabilities,
                config.max_total_shots,
                row.trial_seed,
                MisContinuousRejectsVariant::Aggressive,
            )?;
            let cc = scenario.inclusion_probabilities[conservative.selected_vertex] == best_mean;
            let ac = scenario.inclusion_probabilities[aggressive.selected_vertex] == best_mean;
            cr_c_correct += if cc { 1 } else { 0 };
            cr_a_correct += if ac { 1 } else { 0 };
            cr_c_total += u128::from(conservative.total_observations);
            cr_a_total += u128::from(aggressive.total_observations);
            cr_c_eliminated += conservative.eliminated_vertices_order.len() as u64;
            cr_a_eliminated += aggressive.eliminated_vertices_order.len() as u64;

            let values = [
                "trial".to_owned(),
                scenario.name.clone(),
                row.trial_index.to_string(),
                row.trial_seed.to_string(),
                scenario.inclusion_probabilities.iter().map(|p| format!("{p:.6}")).collect::<Vec<_>>().join(";"),
                row.true_best_vertex.to_string(),
                opt_usize(row.adaptive_selected_vertex),
                opt_bool(row.adaptive_correct),
                row.adaptive_total_shots.to_string(),
                row.uniform_selected_vertex.to_string(),
                row.uniform_correct.to_string(),
                row.uniform_total_shots.to_string(),
                row.successive_rejects_selected_vertex.to_string(),
                row.successive_rejects_correct.to_string(),
                row.successive_rejects_total_shots.to_string(),
                conservative.selected_vertex.to_string(),
                cc.to_string(),
                conservative.total_observations.to_string(),
                join_usize(&conservative.eliminated_vertices_order),
                aggressive.selected_vertex.to_string(),
                ac.to_string(),
                aggressive.total_observations.to_string(),
                join_usize(&aggressive.eliminated_vertices_order),
            ];
            println!("{}", values.iter().map(|v| csv_cell(v)).collect::<Vec<_>>().join(","));
        }

        let summary = aggregate_mis_sampling_trials(&rows)?;
        let (coverage_lo, coverage_hi) = wilson(summary.adaptive_selected, summary.trial_count);
        let (adaptive_err_lo, adaptive_err_hi) = wilson(summary.adaptive_incorrect, summary.adaptive_selected);
        let (uniform_lo, uniform_hi) = wilson(summary.uniform_correct, summary.trial_count);
        let (sr_lo, sr_hi) = wilson(summary.successive_rejects_correct, summary.trial_count);
        let (cr_c_lo, cr_c_hi) = wilson(cr_c_correct, summary.trial_count);
        let (cr_a_lo, cr_a_hi) = wilson(cr_a_correct, summary.trial_count);

        eprintln!(
            "summary,scenario={},n={},adaptive_coverage={:.6},adaptive_coverage_wilson95=[{};{}],adaptive_selective_error={:.6},adaptive_error_wilson95=[{};{}],adaptive_mean_observations={:.3},uniform_accuracy={:.6},uniform_wilson95=[{};{}],sr_accuracy={:.6},sr_wilson95=[{};{}],sr_mean_observations={:.3},cr_c_accuracy={:.6},cr_c_wilson95=[{};{}],cr_c_mean_observations={:.3},cr_c_mean_eliminations={:.3},cr_a_accuracy={:.6},cr_a_wilson95=[{};{}],cr_a_mean_observations={:.3},cr_a_mean_eliminations={:.3}",
            csv_cell(&scenario.name),
            summary.trial_count,
            summary.adaptive_coverage(), fmt_opt(coverage_lo), fmt_opt(coverage_hi),
            summary.adaptive_selective_error(), fmt_opt(adaptive_err_lo), fmt_opt(adaptive_err_hi),
            summary.adaptive_mean_shots(),
            summary.uniform_accuracy(), fmt_opt(uniform_lo), fmt_opt(uniform_hi),
            summary.successive_rejects_accuracy(), fmt_opt(sr_lo), fmt_opt(sr_hi),
            summary.successive_rejects_mean_shots(),
            cr_c_correct as f64 / summary.trial_count as f64, fmt_opt(cr_c_lo), fmt_opt(cr_c_hi),
            cr_c_total as f64 / summary.trial_count as f64,
            cr_c_eliminated as f64 / summary.trial_count as f64,
            cr_a_correct as f64 / summary.trial_count as f64, fmt_opt(cr_a_lo), fmt_opt(cr_a_hi),
            cr_a_total as f64 / summary.trial_count as f64,
            cr_a_eliminated as f64 / summary.trial_count as f64,
        );
    }
    Ok(())
}

fn wilson(successes: u64, total: u64) -> (Option<f64>, Option<f64>) {
    if total == 0 { return (None, None); }
    let n = total as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let den = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / den;
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / den;
    (Some((center - half).max(0.0)), Some((center + half).min(1.0)))
}

fn fmt_opt(x: Option<f64>) -> String {
    x.map(|v| format!("{v:.6}")).unwrap_or_else(|| "NA".to_owned())
}
fn opt_usize(x: Option<usize>) -> String { x.map(|v| v.to_string()).unwrap_or_default() }
fn opt_bool(x: Option<bool>) -> String { x.map(|v| v.to_string()).unwrap_or_default() }
fn join_usize(xs: &[usize]) -> String { xs.iter().map(usize::to_string).collect::<Vec<_>>().join(";") }
fn csv_cell(x: &str) -> String {
    if x.contains(',') || x.contains('"') || x.contains('\n') || x.contains('\r') {
        format!("\"{}\"", x.replace('"', "\"\""))
    } else { x.to_owned() }
}
