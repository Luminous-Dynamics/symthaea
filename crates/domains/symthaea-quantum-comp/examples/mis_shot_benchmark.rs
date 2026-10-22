//! Run a deterministic, synthetic adaptive-vs-uniform sampling benchmark.
//!
//! Usage:
//! cargo run -p symthaea-quantum-comp --example mis_shot_benchmark -- 2000 20261022 > trials.csv 2> summary.txt
//!
//! The CSV contains one row per trial; stderr contains scenario aggregates and Wilson
//! intervals. This is statistical policy validation only, not a QPU or MIS solve.

use std::env;
use std::error::Error;

use symthaea_quantum_comp::mis_informer::MisShotBudgetConfig;
use symthaea_quantum_comp::mis_shot_simulation::{
    MisSamplingAggregate, MisSamplingScenario, MisSamplingTrial,
    aggregate_mis_sampling_trials, simulate_mis_sampling_scenario,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let trials: usize = args.next().unwrap_or_else(|| "2000".to_owned()).parse()?;
    let master_seed: u64 = args.next().unwrap_or_else(|| "20261022".to_owned()).parse()?;
    if args.next().is_some() {
        return Err("usage: mis_shot_benchmark [trials] [master_seed]".into());
    }

    let config = MisShotBudgetConfig::new(8, 8, 512, 0.05)?;
    let scenarios = vec![
        MisSamplingScenario::new("wide-gap", vec![0.80, 0.20])?,
        MisSamplingScenario::new("moderate-gap", vec![0.60, 0.40])?,
        MisSamplingScenario::new("narrow-gap", vec![0.55, 0.45])?,
        MisSamplingScenario::new("exact-tie", vec![0.50, 0.50])?,
    ];

    println!(
        "record_type,scenario,trial_index,trial_seed,master_seed,probabilities,true_best_vertex,adaptive_selected_vertex,adaptive_correct,adaptive_abstention_reason,adaptive_total_shots,adaptive_shots_by_vertex,adaptive_successes_by_vertex,uniform_selected_vertex,uniform_correct,uniform_total_shots,uniform_shots_by_vertex,uniform_successes_by_vertex"
    );

    eprintln!(
        "config,trials={trials},master_seed={master_seed},initial_shots_per_candidate={},batch_shots_per_candidate={},max_total_shots={},familywise_error_probability={}",
        config.initial_shots_per_candidate,
        config.batch_shots_per_candidate,
        config.max_total_shots,
        config.familywise_error_probability
    );

    for scenario in scenarios {
        let outcomes = simulate_mis_sampling_scenario(&scenario, config, trials, master_seed)?;
        for row in &outcomes {
            print_trial(row, master_seed, &scenario.inclusion_probabilities);
        }
        let summary = aggregate_mis_sampling_trials(&outcomes)?;
        print_summary(&scenario.name, &summary);
    }
    Ok(())
}

fn print_trial(row: &MisSamplingTrial, master_seed: u64, probabilities: &[f64]) {
    let values = [
        "trial".to_owned(),
        row.scenario_name.clone(),
        row.trial_index.to_string(),
        row.trial_seed.to_string(),
        master_seed.to_string(),
        probabilities.iter().map(|p| format!("{p:.6}")).collect::<Vec<_>>().join(";"),
        row.true_best_vertex.to_string(),
        optional_usize(row.adaptive_selected_vertex),
        optional_bool(row.adaptive_correct),
        row.adaptive_abstention_reason.map(|reason| format!("{reason:?}")).unwrap_or_default(),
        row.adaptive_total_shots.to_string(),
        join_u64(&row.adaptive_shots_by_vertex),
        join_u64(&row.adaptive_successes_by_vertex),
        row.uniform_selected_vertex.to_string(),
        row.uniform_correct.to_string(),
        row.uniform_total_shots.to_string(),
        join_u64(&row.uniform_shots_by_vertex),
        join_u64(&row.uniform_successes_by_vertex),
    ];
    println!("{}", values.iter().map(|value| csv_cell(value)).collect::<Vec<_>>().join(","));
}

fn print_summary(scenario: &str, summary: &MisSamplingAggregate) {
    let (coverage_low, coverage_high) =
        wilson_interval(summary.adaptive_selected, summary.trial_count);
    let (selective_low, selective_high) =
        wilson_interval(summary.adaptive_incorrect, summary.adaptive_selected);
    let (uniform_low, uniform_high) =
        wilson_interval(summary.uniform_correct, summary.trial_count);
    eprintln!(
        "summary,scenario={},trials={},adaptive_selected={},adaptive_abstained={},adaptive_correct={},adaptive_incorrect={},adaptive_coverage={:.6},adaptive_coverage_wilson95=[{};{}],adaptive_selective_error={:.6},adaptive_selective_error_wilson95=[{};{}],adaptive_mean_shots={:.3},adaptive_mean_shots_fraction_of_fixed={:.6},uniform_correct={},uniform_accuracy={:.6},uniform_accuracy_wilson95=[{};{}],uniform_mean_shots={:.3}",
        csv_cell(scenario),
        summary.trial_count,
        summary.adaptive_selected,
        summary.adaptive_abstained,
        summary.adaptive_correct,
        summary.adaptive_incorrect,
        summary.adaptive_coverage(),
        fmt_optional(coverage_low),
        fmt_optional(coverage_high),
        summary.adaptive_selective_error(),
        fmt_optional(selective_low),
        fmt_optional(selective_high),
        summary.adaptive_mean_shots(),
        summary.adaptive_mean_shots() / summary.uniform_mean_shots(),
        summary.uniform_correct,
        summary.uniform_accuracy(),
        fmt_optional(uniform_low),
        fmt_optional(uniform_high),
        summary.uniform_mean_shots(),
    );
}

fn wilson_interval(successes: u64, total: u64) -> (Option<f64>, Option<f64>) {
    if total == 0 {
        return (None, None);
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    (Some((center - half_width).max(0.0)), Some((center + half_width).min(1.0)))
}

fn join_u64(values: &[u64]) -> String {
    values.iter().map(u64::to_string).collect::<Vec<_>>().join(";")
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn optional_bool(value: Option<bool>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn fmt_optional(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.6}")).unwrap_or_else(|| "NA".to_owned())
}

fn csv_cell(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}
