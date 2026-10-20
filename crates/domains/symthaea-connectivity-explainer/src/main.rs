use std::env;
use std::fs;
use std::process::ExitCode;

use serde_json::Value;
use symthaea_connectivity_explainer::explain_report;

const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;

fn read_json(path: &str) -> Result<Value, String> {
    let metadata = fs::metadata(path).map_err(|error| format!("{path}: {error}"))?;
    if metadata.len() > MAX_INPUT_BYTES {
        return Err(format!("{path}: input exceeds {MAX_INPUT_BYTES} bytes"));
    }
    let bytes = fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        return Err(format!(
            "usage: {} <multiwindow-report.json> [output.json]",
            args.first()
                .map(String::as_str)
                .unwrap_or("symthaea-connectivity-explain")
        ));
    }

    let input = read_json(&args[1])?;
    let explanation = explain_report(&input)
        .map_err(|error| format!("explanation refused: {error}"))?;
    let output = serde_json::to_vec_pretty(&explanation)
        .map_err(|error| format!("cannot serialize explanation: {error}"))?;

    if let Some(path) = args.get(2) {
        fs::write(path, output).map_err(|error| format!("cannot write {path}: {error}"))?;
        println!("Evidence-grounded explanation written to {path}");
    } else {
        println!(
            "{}",
            String::from_utf8(output).map_err(|error| error.to_string())?
        );
    }
    println!("Template-driven explanation only; source evidence class and limitations are preserved.");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
