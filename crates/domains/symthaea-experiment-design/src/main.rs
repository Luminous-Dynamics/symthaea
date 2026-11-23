use std::io::{self, Read};
use std::process::ExitCode;

use serde::Serialize;
use symthaea_experiment_design::{
    ExperimentDesignInput, covariance::{
        covariance_redundancy_fixture, rank_covariance_aware_experiments,
    }, galileo_demo_input, rank_experiments,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() > 1 {
        print_usage();
        return ExitCode::from(2);
    }

    match args.first().map(String::as_str) {
        Some("--help") => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some("--demo") => write_json(&match rank_experiments(&galileo_demo_input()) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("experiment design rejected: {error}");
                return ExitCode::FAILURE;
            }
        }),
        Some("--covariance-demo") => {
            let report = match rank_covariance_aware_experiments(&covariance_redundancy_fixture()) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("covariance-aware experiment design rejected: {error}");
                    return ExitCode::FAILURE;
                }
            };
            write_json(&report)
        }
        Some(unknown) => {
            eprintln!("unknown argument: {unknown}");
            print_usage();
            ExitCode::from(2)
        }
        None => {
            let mut text = String::new();
            if let Err(error) = io::stdin().read_to_string(&mut text) {
                eprintln!("failed to read input JSON: {error}");
                return ExitCode::FAILURE;
            }
            let input = match serde_json::from_str::<ExperimentDesignInput>(&text) {
                Ok(input) => input,
                Err(error) => {
                    eprintln!("invalid experiment design JSON: {error}");
                    return ExitCode::from(2);
                }
            };
            let report = match rank_experiments(&input) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("experiment design rejected: {error}");
                    return ExitCode::FAILURE;
                }
            };
            write_json(&report)
        }
    }
}

fn write_json<T: Serialize>(value: &T) -> ExitCode {
    match serde_json::to_writer_pretty(io::stdout().lock(), value) {
        Ok(()) => {
            println!();
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("failed to serialize experiment design report: {error}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("symthaea-experiment-design");
    println!("Ranks candidate interventions from explicit rival-model predictions.");
    println!("  --demo              run the bundled synthetic Galileo screening fixture");
    println!("  --covariance-demo   run the synthetic covariance/redundancy known-answer control");
    println!("  --help              show this help");
    println!("  no args             read a physics-experiment-design-v0.1 JSON request from stdin");
    println!("The covariance score is a prior-free screening statistic, not expected information gain.");
}
