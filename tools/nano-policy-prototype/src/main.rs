use std::io::{self, Read};
use std::process::ExitCode;

use nano_policy_prototype::{select, Request};

fn main() -> ExitCode {
    let mut input = String::new();
    if let Err(error) = io::stdin().read_to_string(&mut input) {
        eprintln!("failed to read JSON request from stdin: {error}");
        return ExitCode::FAILURE;
    }

    let request: Request = match serde_json::from_str(&input) {
        Ok(request) => request,
        Err(error) => {
            eprintln!("invalid nano-policy request: {error}");
            return ExitCode::FAILURE;
        }
    };

    match select(&request) {
        Ok(response) => match serde_json::to_string_pretty(&response) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("failed to serialize response: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("request rejected by fail-closed validation: {error}");
            ExitCode::FAILURE
        }
    }
}
