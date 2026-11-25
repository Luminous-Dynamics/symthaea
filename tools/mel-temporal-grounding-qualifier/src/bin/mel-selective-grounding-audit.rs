//! CLI for selective temporal-grounding and reason-specific abstention audits.

use std::env;
use std::fs;
use std::io::{self, Read};
use std::process::ExitCode;

use mel_temporal_grounding_qualifier::temporal_selective::audit_selective_batch_json;

fn read_input() -> Result<String, String> {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        return Err("usage: mel-selective-grounding-audit <batch.json|->".into());
    };
    if args.next().is_some() {
        return Err("expected exactly one input path".into());
    }
    if path == "-" {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|error| format!("failed reading stdin: {error}"))?;
        Ok(input)
    } else {
        fs::read_to_string(&path)
            .map_err(|error| format!("failed reading {}: {error}", path.to_string_lossy()))
    }
}

fn run() -> Result<String, String> {
    let input = read_input()?;
    let report = audit_selective_batch_json(&input)?;
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed serializing selective-grounding report: {error}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("mel-selective-grounding-audit: {error}");
            ExitCode::from(2)
        }
    }
}
