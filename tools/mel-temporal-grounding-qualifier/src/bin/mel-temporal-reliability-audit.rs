//! CLI for corpus-level temporal music annotation reliability.

use std::env;
use std::fs;
use std::io::{self, Read};
use std::process::ExitCode;

use mel_temporal_grounding_qualifier::temporal_reliability::audit_reliability_corpus_json_with_source_bootstrap;

fn read_input() -> Result<String, String> {
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        return Err("usage: mel-temporal-reliability-audit <corpus.json|->".into());
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
    let report = audit_reliability_corpus_json_with_source_bootstrap(&input, 2_000, 0x2026_1125)?;
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed serializing reliability report: {error}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("mel-temporal-reliability-audit: {error}");
            ExitCode::from(2)
        }
    }
}
