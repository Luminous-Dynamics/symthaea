//! Audit a frozen proposal/simulator-call ledger.
//!
//! Exit status:
//! - 0: exact call budget and source-reported count both reconcile.
//! - 2: input, argument, or ledger-contract error.
//! - 3: the ledger is valid but the declared budget/reconciliation gate failed.
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;
use symthaea_nano_policy::sim_budget::{
    audit_json_ledger, MAX_SIMULATOR_LEDGER_BYTES,
};

const USAGE: &str = "nano-policy-budget-audit --input simulator-call-ledger.json

Audits proposal attempts separately from actual simulator process invocations.
A failed, timed-out, interrupted or non-finite simulator invocation still consumes
one simulator-call budget unit. Preflight rejects consume proposal budget only.

Options:
  --input PATH     Frozen simulator-call JSON ledger (required)
  --help           Print this help

The tool emits a JSON audit report to stdout. Exit code 0 means the run claims
completion, exactly the preregistered number of simulator invocations are accounted
for, every invocation has a terminal receipt, and the original summary call count
matches. Exit code 3 indicates a valid ledger that is incomplete, over budget,
unresolved, or otherwise not a passing equal-budget run. Exit code 2 indicates
malformed input or an invalid ledger contract.

This is a structural ledger audit. It checks digest shapes and counts but does not
dereference receipt artifacts, validate simulator physics, or prove that external
process receipts are truthful.";

fn main() -> ExitCode {
    match run() {
        Ok(exit_code) => exit_code,
        Err(error) => {
            eprintln!("nano-policy-budget-audit: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return if args.is_empty() {
            Err("missing arguments; see --help".to_string())
        } else {
            Ok(ExitCode::SUCCESS)
        };
    }

    let mut input_path: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;
        match flag {
            "--input" => {
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("missing value after {flag}"))?
                    .clone();
                index += 1;
                if input_path.replace(value).is_some() {
                    return Err("--input was supplied more than once".to_string());
                }
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
    if metadata.len() > MAX_SIMULATOR_LEDGER_BYTES as u64 {
        return Err(format!(
            "simulator ledger exceeds the {MAX_SIMULATOR_LEDGER_BYTES}-byte limit"
        ));
    }
    let raw = fs::read(path).map_err(|error| format!("cannot read simulator ledger: {error}"))?;
    if raw.len() > MAX_SIMULATOR_LEDGER_BYTES {
        return Err(format!(
            "simulator ledger exceeds the {MAX_SIMULATOR_LEDGER_BYTES}-byte limit"
        ));
    }

    let audit = audit_json_ledger(&raw)?;
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    serde_json::to_writer_pretty(&mut output, &audit)
        .map_err(|error| format!("cannot serialize audit report: {error}"))?;
    writeln!(output).map_err(|error| format!("cannot write audit report: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("cannot flush audit report: {error}"))?;

    if audit.audit_pass {
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!(
            "nano-policy-budget-audit: budget gate failed: disposition={:?}, reported_count_matches_observed={}",
            audit.disposition,
            audit.reported_count_matches_observed
        );
        Ok(ExitCode::from(3))
    }
}
