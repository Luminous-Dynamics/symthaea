use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

use symthaea_robodojo_contract::{
    compare_paired_trials, summarize_trials, ActionProposal, ActionRejection, AxisLimit,
    EpisodeGuard, GateConfig, ObservationDecision, ObservationFrame, TrialRecord,
};

const TRIALS_HEADER: &str = "task_id\tvariant_id\ttrial_id\tprogress\tcompleted\tvalid_run\ttimed_out\telapsed_seconds\tmodel_calls\taction_steps\testimated_cost_usd\tsafety_events";

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
            println!("task_id\tvariant_id\tattempts\tvalid\tinvalid\tcompletions\tcompletion_all_pct\tcompletion_scored_pct\tmean_progress_pct\tci95_low_pct\tci95_high_pct\tmean_elapsed_s\tmean_model_calls\tmean_action_steps\tmean_cost_usd\ttimeouts\tsafety_events");
            for row in summaries {
                let valid_completion = optional_pct(row.completion_rate_valid_trials);
                let (lo, hi) = match row.progress_ci95 {
                    Some((lo, hi)) => (pct(lo), pct(hi)),
                    None => ("NA".to_owned(), "NA".to_owned()),
                };
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{:.3}\t{:.4}\t{}\t{}",
                    row.task_id, row.variant_id, row.total_trials, row.valid_trials,
                    row.invalid_trials, row.completions, pct(row.completion_rate_all_trials),
                    valid_completion, optional_pct(row.mean_progress), lo, hi,
                    row.mean_elapsed_seconds, row.mean_model_calls, row.mean_action_steps,
                    row.mean_estimated_cost_usd, row.timeout_count, row.safety_event_count
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
            println!("baseline\tcandidate\tpairs\tscored_pairs\ttasks_scored\tprogress_delta_pp\tci95_low_pp\tci95_high_pp\tcompletion_rate_delta_pp\tvalid_rate_delta_pp\telapsed_delta_s\tcost_delta_usd\tsafety_event_delta");
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}",
                comparison.baseline_variant, comparison.candidate_variant,
                comparison.total_pairs, comparison.scored_pairs, comparison.tasks_with_scored_pairs,
                comparison.mean_progress_delta.map(|x| format!("{:.4}", x * 100.0)).unwrap_or_else(|| "NA".to_owned()),
                lo, hi, comparison.completion_rate_delta * 100.0,
                comparison.valid_run_rate_delta * 100.0,
                comparison.mean_elapsed_seconds_delta, comparison.mean_cost_usd_delta,
                comparison.safety_event_count_delta
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
        if columns.len() != 12 {
            return Err(invalid_input(format!(
                "{}:{} expected 12 tab-separated fields, got {}",
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
        let record = TrialRecord {
            task_id: columns[0].to_owned(),
            variant_id: columns[1].to_owned(),
            trial_id: columns[2].to_owned(),
            progress,
            completed: boolean(4, "completed")?,
            valid_run: boolean(5, "valid_run")?,
            timed_out: boolean(6, "timed_out")?,
            elapsed_seconds: number(7, "elapsed_seconds")?,
            model_calls: integer(8, "model_calls")?,
            action_steps: integer(9, "action_steps")?,
            estimated_cost_usd: number(10, "estimated_cost_usd")?,
            safety_events: integer(11, "safety_events")?,
        };
        record.validate().map_err(|error| invalid_input(format!(
            "line {} invalid trial record: {error:?}", zero_based_line + 2
        )))?;
        records.push(record);
    }
    if records.is_empty() { return Err(invalid_input("trial file contains no records").into()); }
    Ok(records)
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
    println!();
    println!("TRIALS.tsv must use the exact header documented in README.md. Use NA progress for attempts that have no valid final grade.");
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

    #[test]
    fn synthetic_rc_fixture_parses_and_summarizes() {
        let records = read_trials(fixture_path()).expect("synthetic fixture must parse");
        assert_eq!(records.len(), 12);
        let summaries = summarize_trials(&records, 200, 42).expect("fixture must summarize");
        assert_eq!(summaries.len(), 4);
        assert!(summaries.iter().all(|row| row.total_trials == 3));
        assert!(summaries.iter().all(|row| row.valid_trials == 2));
        assert!(summaries.iter().all(|row| row.invalid_trials == 1));
        assert!(summaries.iter().all(|row| row.timeout_count == 1));
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
        assert_eq!(comparison.progress_delta_ci95, Some((0.125, 0.25)));
    }
}
