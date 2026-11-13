use ev_soc_stream_baseline::audit_corpus;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut corpus_path = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => corpus_path = args.next(),
            _ => {
                eprintln!("unknown argument: {arg}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(corpus_path) = corpus_path else {
        eprintln!("usage: ev-soc-stream-baseline --corpus PATH");
        return ExitCode::FAILURE;
    };
    let bytes = match fs::read(&corpus_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read corpus {corpus_path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    match audit_corpus(&bytes) {
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
            eprintln!("EV SOC stream audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}
