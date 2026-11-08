//! Independent post-run benchmark metric evaluator.
//!
//! Input is a post-outcome ledger and must never be passed to the candidate selector.
//! The emitted report includes SHA-256 of the exact raw ledger bytes.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use symthaea_nano_policy::metrics::{
    evaluate_json_ledger, DEFAULT_BOOTSTRAP_REPLICATES, MAX_LEDGER_BYTES,
};

const USAGE: &str = "nano-policy-metrics --input outcome-ledger.json [--bootstrap-replicates N] [--seed N]

Evaluates a frozen post-outcome JSON ledger and writes a JSON metrics report to stdout.

Options:
  --input PATH                Post-outcome ledger JSON (required)
  --bootstrap-replicates N    Deterministic group-bootstrap replicates (default 2000)
  --seed N                    Bootstrap seed (default 42)
  --help                      Print this help

The outcome ledger must stay separate from the selector input. The evaluator checks matching
target/group IDs, attempt ranks, failure/loss semantics and provenance digests. It computes
success-at-budget, normalized success AUC, failure/duplicate/cost metrics, and paired group
bootstrap comparisons. It does not execute the selector or scientific oracle.";

fn main() {
    if let Err(error) = run() {
        eprintln!("nano-policy-metrics: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return if args.is_empty() {
            Err("missing arguments; see --help".to_string())
        } else {
            Ok(())
        };
    }

    let mut input_path: Option<String> = None;
    let mut bootstrap_replicates = DEFAULT_BOOTSTRAP_REPLICATES;
    let mut seed = 42_u64;
    let mut seen_bootstrap = false;
    let mut seen_seed = false;
    let mut index = 0;

    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;
        match flag {
            "--input" => {
                let value = next_value(&args, &mut index, flag)?;
                if input_path.replace(value).is_some() {
                    return Err("--input was supplied more than once".to_string());
                }
            }
            "--bootstrap-replicates" => {
                let value = next_value(&args, &mut index, flag)?;
                if seen_bootstrap {
                    return Err("--bootstrap-replicates was supplied more than once".to_string());
                }
                seen_bootstrap = true;
                bootstrap_replicates = value
                    .parse::<usize>()
                    .map_err(|_| "--bootstrap-replicates must be an integer".to_string())?;
            }
            "--seed" => {
                let value = next_value(&args, &mut index, flag)?;
                if seen_seed {
                    return Err("--seed was supplied more than once".to_string());
                }
                seen_seed = true;
                seed = value
                    .parse::<u64>()
                    .map_err(|_| "--seed must be an unsigned 64-bit integer".to_string())?;
            }
            _ => return Err(format!("unknown argument {flag:?}; see --help")),
        }
    }

    let input_path = input_path.ok_or_else(|| "--input is required".to_string())?;
    let path = Path::new(&input_path);
    let metadata =
        fs::metadata(path).map_err(|error| format!("cannot inspect input file: {error}"))?;
    if !metadata.is_file() {
        return Err("input path must identify a regular file".to_string());
    }
    if metadata.len() > MAX_LEDGER_BYTES as u64 {
        return Err(format!(
            "input ledger exceeds the {MAX_LEDGER_BYTES}-byte limit"
        ));
    }
    let raw = fs::read(path).map_err(|error| format!("cannot read input ledger: {error}"))?;
    if raw.len() > MAX_LEDGER_BYTES {
        return Err(format!(
            "input ledger exceeds the {MAX_LEDGER_BYTES}-byte limit"
        ));
    }

    let report = evaluate_json_ledger(&raw, bootstrap_replicates, seed)?;
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    serde_json::to_writer_pretty(&mut output, &report)
        .map_err(|error| format!("cannot serialize report: {error}"))?;
    writeln!(output).map_err(|error| format!("cannot write report: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("cannot flush report: {error}"))?;
    Ok(())
}

fn next_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    let value = args
        .get(*index)
        .ok_or_else(|| format!("missing value after {flag}"))?
        .clone();
    *index += 1;
    Ok(value)
}
