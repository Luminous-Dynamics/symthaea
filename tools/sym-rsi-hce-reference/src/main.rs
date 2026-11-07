use std::env;
use std::fs;
use std::process::ExitCode;

use sym_rsi_hce_reference::{analyze_worker_trace, HceEpoch, WorkerTrace};

fn validate_synthetic_fixtures() -> Result<serde_json::Value, String> {
    let hce = HceEpoch::parse(include_str!("../fixtures/hce-epoch-v0.json"))
        .map_err(|error| format!("HCE fixture JSON: {error}"))?;
    let hce_report = hce.validate().map_err(|error| format!("HCE epoch contract: {error}"))?;

    let trace: WorkerTrace = serde_json::from_str(include_str!("../fixtures/worker-trace-v0.json"))
        .map_err(|error| format!("worker trace fixture JSON: {error}"))?;
    let trace_report = analyze_worker_trace(&trace)
        .map_err(|error| format!("worker trace contract: {error}"))?;

    Ok(serde_json::json!({
        "valid": true,
        "hce": hce_report,
        "worker_trace": trace_report,
        "claim_limit": "Synthetic contract and trace analysis only; no workers were scheduled, no model was run, no hidden labels were accessed, and no research result was measured."
    }))
}

fn main() -> ExitCode {
    let Some(argument) = env::args().nth(1) else {
        eprintln!("usage: sym-rsi-hce-reference --fixtures");
        return ExitCode::from(64);
    };

    if argument != "--fixtures" {
        eprintln!("unknown command {argument:?}; expected --fixtures");
        return ExitCode::from(64);
    }

    match validate_synthetic_fixtures() {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).expect("serializable report"));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("HCE/worker trace fixture contract failed: {error}");
            ExitCode::from(2)
        }
    }
}
