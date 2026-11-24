use proof_plan_contract::scope_equivalence::{parse_dataset, validation_reports};
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: scope-equivalence <dataset.json>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: scope-equivalence <dataset.json>");
        return ExitCode::from(2);
    }

    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            return ExitCode::from(1);
        }
    };
    let dataset = match parse_dataset(&input) {
        Ok(dataset) => dataset,
        Err(error) => {
            eprintln!("SCOPE_EQUIVALENCE_DATASET_INVALID: {error}");
            return ExitCode::from(1);
        }
    };
    let reports = match validation_reports(&dataset) {
        Ok(reports) => reports,
        Err(error) => {
            eprintln!("SCOPE_EQUIVALENCE_DATASET_INVALID: {error}");
            return ExitCode::from(1);
        }
    };

    for report in &reports {
        println!(
            "SCOPE_EQUIVALENCE_PAIR_VALID pair_id={} relation={:?} raw_same={} normalized_same={}",
            report.pair_id,
            report.relation,
            report.base_raw_sha256 == report.variant_raw_sha256,
            report.base_normalized_sha256 == report.variant_normalized_sha256
        );
    }
    println!("SCOPE_EQUIVALENCE_DATASET_VALID pairs={}", reports.len());
    ExitCode::SUCCESS
}
