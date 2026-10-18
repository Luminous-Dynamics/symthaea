use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

use symthaea_robodojo_contract::{
    calibration::{summarize_calibration, SuccessForecast},
    shift::{audit_shift_robustness, ShiftAssignment, ShiftFamily, ShiftSeverity, SHIFT_PLAN_HEADER},
    compare_paired_trials, summarize_trials, ActionProposal, ActionRejection, AxisLimit,
    EpisodeGuard, FailureClass, GateConfig, ObservationDecision, ObservationFrame, TrialRecord,
};

const TRIALS_HEADER: &str = "task_id\tvariant_id\ttrial_id\tprogress\tcompleted\tvalid_run\ttimed_out\tfailure_class\telapsed_seconds\tmodel_calls\taction_steps\testimated_cost_usd\tsafety_events";
const FORECASTS_HEADER: &str = "task_id\tvariant_id\ttrial_id\tpredicted_success_probability";

#[derive(Debug, Clone, PartialEq)]
struct ForecastProbability {
    task_id: String,
    variant_id: String,
    trial_id: String,
    probability: f64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => run_contract_smoke(),
        Some("--help" | "-h") => {
            print_help();
            Ok(())
        }
        Some("--summarize") => {
            if !(2..=4).contains(&args.len()) {
                return Err(invalid_input("usage: --summarize TRIALS.tsv [BOOTSTRAPS] [SEED]").into());
            }
            let records = read_trials(&args[1])?;
            let replicates = parse_optional(&args, 2, 2_000usize, "bootstrap count")?;
            let seed = parse_optional(&args, 3, 42u64, "seed")?;
            let summaries = summarize_trials(&records, replicates, seed)
                .map_err(|error| invalid_input(format!("cannot summarize trials: {error:?}")))?;
            println!("task_id\tvariant_id\tattempts\tvalid\tinvalid\tcompletions\tcompletion_all_pct\tcompletion_scored_pct\tmean_progress_pct\tci95_low_pct\tci95_high_pct\tmean_elapsed_s\tmean_model_calls\tmean_action_steps\tmean_cost_usd\ttimeouts\tthermal_stops\tserver_errors\tinvalid_actions\tsafety_stops\tother_failures\tsafety_events");
            for row in summaries {
                let valid_completion = optional_pct(row.completion_rate_valid_trials);
                let (lo, hi) = match row.progress_ci95 {
                    Some((lo, hi)) => (pct(lo), pct(hi)),
                    None => ("NA".to_owned(), "NA".to_owned()),
                };
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{:.3}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    row.task_id, row.variant_id, row.total_trials, row.valid_trials,
                    row.invalid_trials, row.completions, pct(row.completion_rate_all_trials),
                    valid_completion, optional_pct(row.mean_progress), lo, hi,
                    row.mean_elapsed_seconds, row.mean_model_calls, row.mean_action_steps,
                    row.mean_estimated_cost_usd, row.timeout_count, row.thermal_stop_count,
                    row.server_error_count, row.invalid_action_count, row.safety_stop_count,
                    row.other_failure_count, row.safety_event_count
                );
            }
            Ok(())
        }
        Some("--calibrate") => {
            if !(3..=5).contains(&args.len()) {
                return Err(invalid_input("usage: --calibrate FORECASTS.tsv TRIALS.tsv [BIN_COUNT] [HIGH_CONFIDENCE_THRESHOLD]").into());
            }
            let probabilities = read_forecast_probabilities(&args[1])?;
            let trials = read_trials(&args[2])?;
            let forecasts = join_forecasts(probabilities, &trials)?;
            let bins = parse_optional(&args, 3, 10usize, "calibration bin count")?;
            let threshold = parse_optional(&args, 4, 0.8f64, "high-confidence threshold")?;
            let summaries = summarize_calibration(&forecasts, bins, threshold)
                .map_err(|error| invalid_input(format!("cannot summarize forecasts: {error:?}")))?;
            println!("task_id\tvariant_id\ttrials\tvalid_runs\tsuccesses\tfailures\tobserved_success_pct\tmean_predicted_pct\tbrier_score\tlog_loss\tece\thigh_confidence_trials\thigh_confidence_failure_pct\trisk_coverage_auc");
            for row in summaries {
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.6}",
                    row.task_id, row.variant_id, row.total_trials, row.valid_runs,
                    row.successes, row.failures, pct(row.observed_success_rate),
                    pct(row.mean_predicted_probability), row.brier_score, row.log_loss,
                    row.expected_calibration_error, row.high_confidence_trials,
                    optional_pct(row.high_confidence_failure_rate), row.risk_coverage_auc
                );
            }
            Ok(())
        }
        Some("--shift-audit") => {
            if !(5..=7).contains(&args.len()) {
                return Err(invalid_input("usage: --shift-audit TRIALS.tsv SHIFT_PLAN.tsv BASELINE CANDIDATE [BOOTSTRAPS] [SEED]").into());
            }
            let records = read_trials(&args[1])?;
            let assignments = read_shift_plan(&args[2])?;
            let baseline_name = &args[3];
            let candidate_name = &args[4];
            let replicates = parse_optional(&args, 5, 2_000usize, "bootstrap count")?;
            let seed = parse_optional(&args, 6, 42u64, "seed")?;
            let unexpected: Vec<&str> = records.iter()
                .filter(|r| r.variant_id.as_str() != baseline_name.as_str() && r.variant_id.as_str() != candidate_name.as_str())
                .map(|r| r.variant_id.as_str()).collect();
            if !unexpected.is_empty() {
                return Err(invalid_input(format!("unexpected variants in input: {}", unexpected.join(","))).into());
            }
            let baseline: Vec<TrialRecord> = records.iter()
                .filter(|r| r.variant_id.as_str() == baseline_name.as_str()).cloned().collect();
            let candidate: Vec<TrialRecord> = records.iter()
                .filter(|r| r.variant_id.as_str() == candidate_name.as_str()).cloned().collect();
            let summaries = audit_shift_robustness(&baseline, &candidate, &assignments, replicates, seed)
                .map_err(|error| invalid_input(format!("cannot audit shift strata: {error:?}")))?;
            println!("family\tseverity\tperturbation_spec_sha256\tconditions\tpairs\tscored_pairs\tprogress_delta_pp\tci95_low_pp\tci95_high_pp\tcompletion_rate_delta_pp\tvalid_rate_delta_pp\telapsed_delta_s\tcost_delta_usd\tsafety_event_delta\ttimeout_delta\tthermal_stop_delta\tserver_error_delta\tinvalid_action_delta\tsafety_stop_delta\tother_failure_delta");
            for row in summaries {
                let (lo, hi) = match row.progress_delta_ci95 {
                    Some((lo, hi)) => (format!("{:.4}", lo * 100.0), format!("{:.4}", hi * 100.0)),
                    None => ("NA".to_owned(), "NA".to_owned()),
                };
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    row.family.as_str(), row.severity.as_str(), row.perturbation_spec_sha256,
                    row.condition_count, row.pairs, row.scored_pairs,
                    row.mean_progress_delta.map(|x| format!("{:.4}", x * 100.0)).unwrap_or_else(|| "NA".to_owned()),
                    lo, hi, row.completion_rate_delta * 100.0, row.valid_run_rate_delta * 100.0,
                    row.mean_elapsed_seconds_delta, row.mean_cost_usd_delta,
                    row.safety_event_count_delta, row.timeout_count_delta, row.thermal_stop_count_delta,
                    row.server_error_count_delta, row.invalid_action_count_delta,
                    row.safety_stop_count_delta, row.other_failure_count_delta
                );
            }
            Ok(())
        }
        Some("--compare") => {
            if !(4..=6).contains(&args.len()) {
                return Err(invalid_input("usage: --compare TRIALS.tsv BASELINE_VARIANT CANDIDATE_VARIANT [BOOTSTRAPS] [SEED]").into());
            }
            let records = read_trials(&args[1])?;
            let baseline_name = &args[2];
            let candidate_name = &args[3];
            let replicates = parse_optional(&args, 4, 2_000usize, "bootstrap count")?;
            let seed = parse_optional(&args, 5, 42u64, "seed")?;
            let unexpected: Vec<&str> = records.iter()
                .filter(|row| row.variant_id.as_str() != baseline_name.as_str() && row.variant_id.as_str() != candidate_name.as_str())
                .map(|row| row.variant_id.as_str())
                .collect();
            if !unexpected.is_empty() {
                return Err(invalid_input(format!(
                    "input contains variants other than selected baseline/candidate: {}",
                    unexpected.join(",")
                )).into());
            }
            let baseline: Vec<TrialRecord> = records.iter()
                .filter(|row| row.variant_id.as_str() == baseline_name.as_str()).cloned().collect();
            let candidate: Vec<TrialRecord> = records.iter()
                .filter(|row| row.variant_id.as_str() == candidate_name.as_str()).cloned().collect();
            let comparison = compare_paired_trials(&baseline, &candidate, replicates, seed)
                .map_err(|error| invalid_input(format!("cannot compare paired trials: {error:?}")))?;
            let (lo, hi) = match comparison.progress_delta_ci95 {
                Some((lo, hi)) => (pct(lo), pct(hi)),
                None => ("NA".to_owned(), "NA".to_owned()),
            };
            println!("baseline\tcandidate\tpairs\tscored_pairs\ttasks_scored\tprogress_delta_pp\tci95_low_pp\tci95_high_pp\tcompletion_rate_delta_pp\tvalid_rate_delta_pp\telapsed_delta_s\tcost_delta_usd\tsafety_event_delta\ttimeout_delta\tthermal_stop_delta\tserver_error_delta\tinvalid_action_delta\tsafety_stop_delta\tother_failure_delta");
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                comparison.baseline_variant, comparison.candidate_variant,
                comparison.total_pairs, comparison.scored_pairs, comparison.tasks_with_scored_pairs,
                comparison.mean_progress_delta.map(|x| format!("{:.4}", x * 100.0)).unwrap_or_else(|| "NA".to_owned()),
                lo, hi, comparison.completion_rate_delta * 100.0,
                comparison.valid_run_rate_delta * 100.0,
                comparison.mean_elapsed_seconds_delta, comparison.mean_cost_usd_delta,
                comparison.safety_event_count_delta, comparison.timeout_count_delta,
                comparison.thermal_stop_count_delta, comparison.server_error_count_delta,
                comparison.invalid_action_count_delta, comparison.safety_stop_count_delta,
                comparison.other_failure_count_delta
            );
            Ok(())
        }
        Some(command) => Err(invalid_input(format!("unknown command {command:?}; use --help")).into()),
    }
}

fn parse_optional<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    default: T,
    label: &str,
) -> Result<T, Box<dyn Error>> {
    match args.get(index) {
        Some(raw) => raw.parse::<T>().map_err(|_| invalid_input(format!("invalid {label}: {raw:?}")).into()),
        None => Ok(default),
    }
}

fn read_trials(path: impl AsRef<Path>) -> Result<Vec<TrialRecord>, Box<dyn Error>> {
    let contents = fs::read_to_string(path.as_ref())?;
    let mut lines = contents.lines();
    let header = lines.next().ok_or_else(|| invalid_input("trial file is empty"))?;
    if header.trim_end_matches('\r') != TRIALS_HEADER {
        return Err(invalid_input(format!("unexpected header; expected: {TRIALS_HEADER}")).into());
    }

    let mut records = Vec::new();
    for (zero_based_line, line) in lines.enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') { continue; }
        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() != 13 {
            return Err(invalid_input(format!(
                "{}:{} expected 13 tab-separated fields, got {}",
                path.as_ref().display(), zero_based_line + 2, columns.len()
            )).into());
        }
        let number = |index: usize, name: &str| -> Result<f64, Box<dyn Error>> {
            columns[index].parse::<f64>()
                .map_err(|_| invalid_input(format!("line {} invalid {name}: {:?}", zero_based_line + 2, columns[index])).into())
        };
        let integer = |index: usize, name: &str| -> Result<u32, Box<dyn Error>> {
            columns[index].parse::<u32>()
                .map_err(|_| invalid_input(format!("line {} invalid {name}: {:?}", zero_based_line + 2, columns[index])).into())
        };
        let boolean = |index: usize, name: &str| -> Result<bool, Box<dyn Error>> {
            match columns[index] {
                "true" => Ok(true),
                "false" => Ok(false),
                value => Err(invalid_input(format!("line {} invalid {name}: {value:?}; use true or false", zero_based_line + 2)).into()),
            }
        };
        let progress = match columns[3] {
            "" | "NA" => None,
            raw => Some(raw.parse::<f64>().map_err(|_| invalid_input(format!("line {} invalid progress: {raw:?}", zero_based_line + 2)))?),
        };
        let failure_class = FailureClass::parse(columns[7]).ok_or_else(|| invalid_input(format!(
            "line {} invalid failure_class: {:?}; expected none, timeout, thermal_stop, server_error, invalid_action, safety_stop, or other",
            zero_based_line + 2, columns[7]
        )))?;
        let record = TrialRecord {
            task_id: columns[0].to_owned(),
            variant_id: columns[1].to_owned(),
            trial_id: columns[2].to_owned(),
            progress,
            completed: boolean(4, "completed")?,
            valid_run: boolean(5, "valid_run")?,
            timed_out: boolean(6, "timed_out")?,
            failure_class,
            elapsed_seconds: number(8, "elapsed_seconds")?,
            model_calls: integer(9, "model_calls")?,
            action_steps: integer(10, "action_steps")?,
            estimated_cost_usd: number(11, "estimated_cost_usd")?,
            safety_events: integer(12, "safety_events")?,
        };
        record.validate().map_err(|error| invalid_input(format!(
            "line {} invalid trial record: {error:?}", zero_based_line + 2
        )))?;
        records.push(record);
    }
    if records.is_empty() { return Err(invalid_input("trial file contains no records").into()); }
    Ok(records)
}

fn read_shift_plan(path: impl AsRef<Path>) -> Result<Vec<ShiftAssignment>, Box<dyn Error>> {
    let contents = fs::read_to_string(path.as_ref())?;
    let mut lines = contents.lines();
    let header = lines.next().ok_or_else(|| invalid_input("shift plan is empty"))?;
    if header.trim_end_matches('\r') != SHIFT_PLAN_HEADER {
        return Err(invalid_input(format!("unexpected shift-plan header; expected: {SHIFT_PLAN_HEADER}")).into());
    }
    let mut assignments = Vec::new();
    for (zero_based_line, line) in lines.enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') { continue; }
        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() != 7 {
            return Err(invalid_input(format!("{}:{} expected 7 tab-separated fields, got {}",
                path.as_ref().display(), zero_based_line + 2, columns.len())).into());
        }
        let family = ShiftFamily::parse(columns[3]).ok_or_else(|| invalid_input(format!(
            "line {} unknown shift family: {:?}", zero_based_line + 2, columns[3]
        )))?;
        let severity = ShiftSeverity::parse(columns[4]).ok_or_else(|| invalid_input(format!(
            "line {} unknown shift severity: {:?}", zero_based_line + 2, columns[4]
        )))?;
        let seed = columns[5].parse::<u64>().map_err(|_| invalid_input(format!(
            "line {} invalid seed: {:?}", zero_based_line + 2, columns[5]
        )))?;
        let assignment = ShiftAssignment {
            task_id: columns[0].to_owned(), trial_id: columns[1].to_owned(),
            condition_id: columns[2].to_owned(), family, severity, seed,
            perturbation_spec_sha256: columns[6].to_owned(),
        };
        assignment.validate().map_err(|error| invalid_input(format!(
            "line {} invalid shift assignment: {error:?}", zero_based_line + 2
        )))?;
        assignments.push(assignment);
    }
    if assignments.is_empty() { return Err(invalid_input("shift plan contains no assignments").into()); }
    Ok(assignments)
}

fn read_forecast_probabilities(path: impl AsRef<Path>) -> Result<Vec<ForecastProbability>, Box<dyn Error>> {
    let contents = fs::read_to_string(path.as_ref())?;
    let mut lines = contents.lines();
    let header = lines.next().ok_or_else(|| invalid_input("forecast file is empty"))?;
    if header.trim_end_matches('\r') != FORECASTS_HEADER {
        return Err(invalid_input(format!("unexpected forecast header; expected: {FORECASTS_HEADER}")).into());
    }

    let mut records = Vec::new();
    for (zero_based_line, line) in lines.enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') { continue; }
        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() != 4 {
            return Err(invalid_input(format!(
                "{}:{} expected 4 tab-separated fields, got {}",
                path.as_ref().display(), zero_based_line + 2, columns.len()
            )).into());
        }
        let probability = columns[3].parse::<f64>().map_err(|_| invalid_input(format!(
            "line {} invalid predicted_success_probability: {:?}",
            zero_based_line + 2, columns[3]
        )))?;
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(invalid_input(format!(
                "line {} predicted_success_probability must be finite and in [0, 1]",
                zero_based_line + 2
            )).into());
        }
        records.push(ForecastProbability {
            task_id: columns[0].to_owned(),
            variant_id: columns[1].to_owned(),
            trial_id: columns[2].to_owned(),
            probability,
        });
    }
    if records.is_empty() { return Err(invalid_input("forecast file contains no records").into()); }
    Ok(records)
}

fn join_forecasts(
    probabilities: Vec<ForecastProbability>,
    trials: &[TrialRecord],
) -> Result<Vec<SuccessForecast>, Box<dyn Error>> {
    let mut trial_by_key: BTreeMap<(String, String, String), &TrialRecord> = BTreeMap::new();
    for trial in trials {
        let key = (trial.task_id.clone(), trial.variant_id.clone(), trial.trial_id.clone());
        if trial_by_key.insert(key, trial).is_some() {
            return Err(invalid_input("trial ledger contains duplicate task/variant/trial IDs").into());
        }
    }

    let mut probability_by_key: BTreeMap<(String, String, String), f64> = BTreeMap::new();
    for forecast in probabilities {
        let key = (forecast.task_id, forecast.variant_id, forecast.trial_id);
        if probability_by_key.insert(key, forecast.probability).is_some() {
            return Err(invalid_input("forecast file contains duplicate task/variant/trial IDs").into());
        }
    }

    let trial_keys: BTreeSet<_> = trial_by_key.keys().cloned().collect();
    let forecast_keys: BTreeSet<_> = probability_by_key.keys().cloned().collect();
    if trial_keys != forecast_keys {
        let missing = trial_keys.difference(&forecast_keys).count();
        let extra = forecast_keys.difference(&trial_keys).count();
        return Err(invalid_input(format!(
            "forecast/trial IDs must match exactly; missing forecasts={missing}, unmatched forecasts={extra}"
        )).into());
    }

    let mut joined = Vec::with_capacity(trial_by_key.len());
    for (key, trial) in trial_by_key {
        let probability = probability_by_key.get(&key).expect("key equality checked");
        joined.push(SuccessForecast {
            task_id: trial.task_id.clone(),
            variant_id: trial.variant_id.clone(),
            trial_id: trial.trial_id.clone(),
            predicted_success_probability: *probability,
            success: trial.completed,
            valid_run: trial.valid_run,
        });
    }
    Ok(joined)
}

fn pct(value: f64) -> String { format!("{:.2}", value * 100.0) }

fn optional_pct(value: Option<f64>) -> String {
    value.map(pct).unwrap_or_else(|| "NA".to_owned())
}

fn invalid_input(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidInput, message.into())
}

fn print_help() {
    println!("Standalone RoboDojo-RC contract and trial-analysis harness");
    println!();
    println!("  cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml");
    println!("      Run the deterministic contract smoke trace.");
    println!("  cargo test --locked --manifest-path experiments/robodojo-contract/Cargo.toml");
    println!("      Run action/observation and trial-metric unit tests.");
    println!("  cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --summarize TRIALS.tsv [BOOTSTRAPS] [SEED]");
    println!("      Emit task-by-variant score/reliability metrics as TSV.");
    println!("  cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --compare TRIALS.tsv BASELINE CANDIDATE [BOOTSTRAPS] [SEED]");
    println!("      Compare matched trials; report task-macro partial-progress delta and reliability/cost deltas.");
    println!("  cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --calibrate FORECASTS.tsv TRIALS.tsv [BIN_COUNT] [HIGH_CONFIDENCE_THRESHOLD]");
    println!("  cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --shift-audit TRIALS.tsv SHIFT_PLAN.tsv BASELINE CANDIDATE [BOOTSTRAPS] [SEED]");
    println!("      Score pre-trial completion-probability forecasts with Brier, log loss, ECE, high-confidence failure rate, and risk-coverage AUC.");
    println!("      Audit matched outcomes per shift family, severity, and perturbation configuration digest.");
    println!();
    println!("TRIALS.tsv must use the exact header documented in README.md. FORECASTS.tsv contains only pre-trial probabilities; outcomes are joined from TRIALS.tsv by task/variant/trial ID. SHIFT_PLAN.tsv must match selected task/trial pairs exactly.");
}

fn run_contract_smoke() -> Result<(), Box<dyn Error>> {
    let config = GateConfig::new(
        3,
        vec![
            AxisLimit::new(-1.0, 1.0)
                .map_err(|error| invalid_input(format!("invalid smoke axis limit: {error:?}")))?,
            AxisLimit::new(0.0, 0.08)
                .map_err(|error| invalid_input(format!("invalid smoke axis limit: {error:?}")))?,
        ],
        0.5,
        3,
    ).map_err(|error| invalid_input(format!("invalid smoke configuration: {error:?}")))?;

    let mut gate = EpisodeGuard::new(config);
    gate.start_episode("contract-smoke")
        .map_err(|error| invalid_input(format!("invalid smoke episode: {error:?}")))?;

    let frame = |step, prediction_error| ObservationFrame {
        episode_id: "contract-smoke".to_owned(),
        step,
        state_values: vec![0.1, 0.2, 0.3],
        prediction_error,
    };
    let action = |step| ActionProposal {
        episode_id: "contract-smoke".to_owned(),
        based_on_step: step,
        values: vec![0.25, 0.04],
    };

    assert_eq!(gate.observe(&frame(0, 0.1)), ObservationDecision::Accepted);
    assert_eq!(gate.approve_action(&action(0)), Ok(vec![0.25, 0.04]));

    assert!(matches!(gate.observe(&frame(1, 0.7)), ObservationDecision::RequestReplan { consecutive_high_error: 1, .. }));
    assert_eq!(gate.approve_action(&action(1)), Err(ActionRejection::ReplanRequired { observed_step: 1 }));
    assert!(matches!(gate.observe(&frame(2, 0.8)), ObservationDecision::RequestReplan { consecutive_high_error: 2, .. }));
    assert!(matches!(gate.observe(&frame(3, 0.9)), ObservationDecision::SafeStop(_)));
    assert!(gate.stopped().is_some());

    println!("contract_smoke=PASS accepted_actions=1 replan_requests=2 stop_latched=1 official_robodojo_result=false");
    Ok(())
}


#[cfg(test)]
mod cli_tests {
    use super::*;

    fn fixture_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("synthetic_trials.tsv")
    }

    fn forecast_fixture_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("synthetic_forecasts.tsv")
    }

    #[test]
    fn synthetic_forecasts_produce_calibration_and_risk_coverage_metrics() {
        let probabilities = read_forecast_probabilities(forecast_fixture_path())
            .expect("synthetic forecasts must parse");
        assert_eq!(probabilities.len(), 12);
        let trials = read_trials(fixture_path()).expect("synthetic trial ledger must parse");
        let forecasts = join_forecasts(probabilities, &trials)
            .expect("probabilities must join to authoritative trial outcomes");
        let summaries = summarize_calibration(&forecasts, 10, 0.8).expect("forecasts must summarize");
        assert_eq!(summaries.len(), 4);
        assert!(summaries.iter().all(|row| row.total_trials == 3));
        assert!(summaries.iter().all(|row| row.valid_runs == 2));
        assert!(summaries.iter().all(|row| row.risk_coverage.len() == 3));
        assert!(summaries.iter().all(|row| row.brier_score.is_finite()));
        assert!(summaries.iter().all(|row| row.log_loss.is_finite()));
        assert!(summaries.iter().all(|row| row.expected_calibration_error.is_finite()));
    }

    #[test]
    fn forecast_join_rejects_missing_or_duplicate_ids() {
        let probabilities = read_forecast_probabilities(forecast_fixture_path())
            .expect("synthetic forecasts must parse");
        let trials = read_trials(fixture_path()).expect("synthetic trial ledger must parse");
        let mut missing = probabilities.clone();
        missing.pop();
        assert!(join_forecasts(missing, &trials).is_err());
        let duplicated = vec![probabilities[0].clone(), probabilities[0].clone()];
        assert!(join_forecasts(duplicated, &trials).is_err());
    }

    #[test]
    fn synthetic_rc_fixture_parses_and_summarizes() {
        let records = read_trials(fixture_path()).expect("synthetic fixture must parse");
        assert_eq!(records.len(), 12);
        let summaries = summarize_trials(&records, 200, 42).expect("fixture must summarize");
        assert_eq!(summaries.len(), 4);
        assert!(summaries.iter().all(|row| row.total_trials == 3));
        assert!(summaries.iter().all(|row| row.valid_trials == 2));
        assert!(summaries.iter().all(|row| row.invalid_trials == 1));
        assert_eq!(summaries.iter().map(|row| row.timeout_count).sum::<usize>(), 1);
        assert_eq!(summaries.iter().map(|row| row.thermal_stop_count).sum::<usize>(), 1);
        assert_eq!(summaries.iter().map(|row| row.server_error_count).sum::<usize>(), 1);
        assert_eq!(summaries.iter().map(|row| row.safety_stop_count).sum::<usize>(), 1);
    }

    #[test]
    fn synthetic_fixture_paired_comparison_keeps_invalid_trials_in_reliability_delta() {
        let records = read_trials(fixture_path()).expect("synthetic fixture must parse");
        let baseline: Vec<_> = records.iter()
            .filter(|row| row.variant_id == "baseline").cloned().collect();
        let candidate: Vec<_> = records.iter()
            .filter(|row| row.variant_id == "symthaea_sidecar").cloned().collect();
        let comparison = compare_paired_trials(&baseline, &candidate, 200, 42)
            .expect("fixture variants are paired");
        assert_eq!(comparison.total_pairs, 6);
        assert_eq!(comparison.scored_pairs, 4);
        assert_eq!(comparison.tasks_with_scored_pairs, 2);
        assert_eq!(comparison.valid_run_rate_delta, 0.0);
        assert_eq!(comparison.timeout_count_delta, -1);
        assert_eq!(comparison.thermal_stop_count_delta, -1);
        assert_eq!(comparison.server_error_count_delta, 1);
        assert_eq!(comparison.safety_stop_count_delta, 1);
        assert_eq!(comparison.mean_progress_delta, Some(0.3125));
        let (lower, upper) = comparison.progress_delta_ci95.expect("four scored pairs yield a bootstrap interval");
        assert!(lower >= 0.25 && upper <= 0.375 && lower <= upper);
    }
}
