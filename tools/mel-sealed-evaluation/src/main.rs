use std::env;
use std::fs;
use std::process::ExitCode;

use mel_sealed_evaluation::evaluate;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!(
            "usage: mel-sealed-evaluation MANIFEST.json PREDICTIONS.json HUMAN-REFERENCE.json"
        );
        return ExitCode::FAILURE;
    }

    let mut inputs = Vec::with_capacity(3);
    for path in &args {
        match fs::read_to_string(path) {
            Ok(contents) => inputs.push(contents),
            Err(error) => {
                eprintln!("failed to read {path}: {error}");
                return ExitCode::FAILURE;
            }
        }
    }

    let report = match evaluate(&inputs[0], &inputs[1], &inputs[2]) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("sealed evaluation rejected: {error}");
            return ExitCode::FAILURE;
        }
    };

    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize evaluation report: {error}");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
