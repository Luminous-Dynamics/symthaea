//! CLI for independently scoring music temporal-grounding annotations.

use std::env;
use std::fs;
use std::io::{self, Read};
use std::process::ExitCode;

use mel_temporal_grounding_qualifier::temporal_consensus::audit_multi_annotator_case_json;

fn read_input() -> Result<String, String> {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        return Err("usage: mel-multi-annotator-grounding-audit <case.json|->".into());
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
    let report = audit_multi_annotator_case_json(&input)?;
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed serializing multi-annotator report: {error}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("mel-multi-annotator-grounding-audit: {error}");
            ExitCode::from(2)
        }
    }
}
