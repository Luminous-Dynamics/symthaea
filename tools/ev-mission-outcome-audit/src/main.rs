use ev_mission_outcome_audit::audit_corpus_with_trust;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut schema_path = None;
    let mut corpus_path = None;
    let mut trust_manifest_path = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--schema" => schema_path = args.next(),
            "--corpus" => corpus_path = args.next(),
            "--trust-manifest" => trust_manifest_path = args.next(),
            _ => {
                eprintln!("unknown argument: {arg}");
                return ExitCode::FAILURE;
            }
        }
    }
    let (Some(schema_path), Some(corpus_path), Some(trust_manifest_path)) =
        (schema_path, corpus_path, trust_manifest_path)
    else {
        eprintln!("usage: ev-mission-outcome-audit --schema PATH --trust-manifest PATH --corpus PATH");
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
    let trust_manifest = match read(&trust_manifest_path) {
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
    match audit_corpus_with_trust(&schema, &trust_manifest, &corpus) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("cannot serialize mission outcome report: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("mission outcome audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}
