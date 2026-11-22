use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use mel_xacle_audit::{
    compare_csv, score_csv, validate_submission_csv, DEFAULT_BOOTSTRAP_SAMPLES,
};

fn write_report(path: &str, report: &impl serde::Serialize) -> Result<(), String> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create report directory: {error}"))?;
        }
    }
    fs::write(path, format!("{json}\n"))
        .map_err(|error| format!("failed to write report {path}: {error}"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("score") if args.len() == 4 || args.len() == 5 => {
            let bootstrap = if args.len() == 5 {
                args[4]
                    .parse::<usize>()
                    .map_err(|_| "bootstrap count must be a positive integer".to_owned())?
            } else {
                DEFAULT_BOOTSTRAP_SAMPLES
            };
            let metadata = fs::read(&args[1])
                .map_err(|error| format!("failed to read metadata {}: {error}", args[1]))?;
            let predictions = fs::read(&args[2])
                .map_err(|error| format!("failed to read predictions {}: {error}", args[2]))?;
            let report = score_csv(&metadata, &predictions, &args[2], bootstrap, 42)?;
            write_report(&args[3], &report)?;
            println!("Scored {} items; report written to {}", report.metrics.count, args[3]);
            Ok(())
        }
        Some("compare") if args.len() == 5 || args.len() == 6 => {
            let bootstrap = if args.len() == 6 {
                args[5]
                    .parse::<usize>()
                    .map_err(|_| "bootstrap count must be a positive integer".to_owned())?
            } else {
                DEFAULT_BOOTSTRAP_SAMPLES
            };
            let metadata = fs::read(&args[1])
                .map_err(|error| format!("failed to read metadata {}: {error}", args[1]))?;
            let baseline = fs::read(&args[2])
                .map_err(|error| format!("failed to read baseline {}: {error}", args[2]))?;
            let candidate = fs::read(&args[3])
                .map_err(|error| format!("failed to read candidate {}: {error}", args[3]))?;
            let report = compare_csv(
                &metadata,
                &baseline,
                &args[2],
                &candidate,
                &args[3],
                bootstrap,
                42,
            )?;
            write_report(&args[4], &report)?;
            println!("Compared {} items; report written to {}", report.item_count, args[4]);
            Ok(())
        }
        Some("validate-submission") if args.len() == 4 => {
            let manifest = fs::read(&args[1])
                .map_err(|error| format!("failed to read evaluation manifest {}: {error}", args[1]))?;
            let predictions = fs::read(&args[2])
                .map_err(|error| format!("failed to read submission {}: {error}", args[2]))?;
            let report = validate_submission_csv(&manifest, &predictions, &args[2])?;
            write_report(&args[3], &report)?;
            println!(
                "Submission covers {} evaluation items; report written to {}",
                report.expected_item_count, args[3]
            );
            Ok(())
        }
        _ => Err(
            "usage:\n  mel-xacle-audit score METADATA.csv PREDICTIONS.csv REPORT.json [BOOTSTRAP_SAMPLES]\n  mel-xacle-audit compare METADATA.csv BASELINE.csv CANDIDATE.csv REPORT.json [BOOTSTRAP_SAMPLES]\n  mel-xacle-audit validate-submission EVALUATION.csv PREDICTIONS.csv REPORT.json"
                .to_owned(),
        ),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mel-xacle-audit: {error}");
            ExitCode::FAILURE
        }
    }
}
