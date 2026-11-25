use std::process::ExitCode;
use stateguard_b0_policy::{decide_json, PolicyInput};

fn main() -> ExitCode {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "policy-input.json".to_owned());
    let input = match std::fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    match decide_json(&input) {
        Ok(decision) => {
            println!("{decision}");
            ExitCode::SUCCESS
        }
        Err(issues) => {
            for issue in issues {
                eprintln!("{}: {}", issue.path, issue.message);
            }
            ExitCode::FAILURE
        }
    }
}
