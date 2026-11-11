use ev_early_window_outcome_audit::audit_corpus;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut schema_path = None;
    let mut corpus_path = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--schema" => schema_path = args.next(),
            "--corpus" => corpus_path = args.next(),
            _ => {
                eprintln!("unknown argument: {arg}");
                return ExitCode::FAILURE;
            }
        }
    }
    let (Some(schema_path), Some(corpus_path)) = (schema_path, corpus_path) else {
        eprintln!("usage: ev-early-window-outcome-audit --schema PATH --corpus PATH");
        return ExitCode::FAILURE;
    };
    let read = |path: &str| match fs::read(path) {
        Ok(bytes) => Ok(bytes),
        Err(error) => Err(format!("cannot read {path}: {error}")),
    };
    let schema = match read(&schema_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let corpus = match read(&corpus_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    match audit_corpus(&schema, &corpus) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("cannot serialize early-window outcome report: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("early-window outcome audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}
