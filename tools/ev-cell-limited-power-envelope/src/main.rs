use ev_cell_limited_power_envelope::audit_corpus;
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
        eprintln!("usage: ev-cell-limited-power-envelope --schema PATH --corpus PATH");
        return ExitCode::FAILURE;
    };
    let schema = match fs::read(&schema_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read schema {schema_path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let corpus = match fs::read(&corpus_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read corpus {corpus_path}: {error}");
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
                eprintln!("cannot serialize report: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("cell-limited power-envelope audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}
