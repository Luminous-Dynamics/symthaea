use std::io::{self, Read};
use std::process::ExitCode;

use symthaea_experiment_design::{ExperimentDesignInput, galileo_demo_input, rank_experiments};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() > 1 {
        eprintln!("usage: physics-experiment-design [--demo|--help] < input.json");
        return ExitCode::from(2);
    }

    let input = match args.first().map(String::as_str) {
        Some("--help") => {
            println!("symthaea-experiment-design");
            println!("Ranks experiments by prior-free, worst-case standardized separation per relative cost.");
            println!("  --demo  run the bundled Galileo constant-acceleration vs constant-velocity fixture");
            println!("  no args: read a JSON ExperimentDesignInput from stdin and write a JSON DesignReport");
            println!("This score is a screening heuristic, not Bayesian expected information gain or experimental proof.");
            return ExitCode::SUCCESS;
        }
        Some("--demo") => galileo_demo_input(),
        Some(unknown) => {
            eprintln!("unknown argument: {unknown}");
            eprintln!("usage: physics-experiment-design [--demo|--help] < input.json");
            return ExitCode::from(2);
        }
        None => {
            let mut text = String::new();
            if let Err(error) = io::stdin().read_to_string(&mut text) {
                eprintln!("failed to read input JSON: {error}");
                return ExitCode::FAILURE;
            }
            match serde_json::from_str::<ExperimentDesignInput>(&text) {
                Ok(input) => input,
                Err(error) => {
                    eprintln!("invalid experiment design JSON: {error}");
                    return ExitCode::from(2);
                }
            }
        }
    };

    let report = match rank_experiments(&input) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("experiment design rejected: {error}");
            return ExitCode::FAILURE;
        }
    };

    match serde_json::to_writer_pretty(io::stdout().lock(), &report) {
        Ok(()) => {
            println!();
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("failed to serialize design report: {error}");
            ExitCode::FAILURE
        }
    }
}
