use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(input_path) = std::env::args().nth(1) else {
        eprintln!("usage: mace-decision-episode-auditor <dataset.json>");
        return ExitCode::from(2);
    };

    let source = match fs::read(&input_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {input_path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    match mace_decision_episode_auditor::audit_json(&source) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("failed to encode audit report: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("dataset rejected: {error}");
            ExitCode::FAILURE
        }
    }
}
