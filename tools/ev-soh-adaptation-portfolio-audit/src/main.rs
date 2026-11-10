use ev_soh_adaptation_portfolio_audit::audit_corpus;
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
        eprintln!("usage: ev-soh-adaptation-portfolio-audit --schema PATH --corpus PATH");
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
                eprintln!("cannot serialize adaptation portfolio report: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("SOH adaptation portfolio audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}
