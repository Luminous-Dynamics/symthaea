use basalt_trace_verifier::{verify_reference, Trace};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut trace_path = None::<String>;
    let mut report_path = None::<String>;
    let mut schema_path = None::<String>;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--trace" => trace_path = args.next(),
            "--out" => report_path = args.next(),
            "--schema" => schema_path = args.next(),
            _ => {
                eprintln!("unknown argument: {arg}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = trace_path else {
        eprintln!("usage: basalt-trace-verifier --trace TRACE.json [--schema SCHEMA.json] [--out REPORT.json]");
        return ExitCode::from(2);
    };
    let raw = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read trace {path}: {error}");
            return ExitCode::from(2);
        }
    };
    let trace: Trace = match serde_json::from_slice(&raw) {
        Ok(value) => value,
        Err(error) => {
            let invalid = json!({
                "report_version": 1,
                "status": "invalid_evidence",
                "input": path,
                "error": error.to_string(),
                "trace_sha256": format!("sha256:{}", hex_lower(&Sha256::digest(&raw)))
            });
            return emit(invalid, report_path, 2);
        }
    };
    let mut report = verify_reference(&trace, &raw);
    if let Some(schema_path) = schema_path {
        match fs::read(&schema_path) {
            Ok(schema_bytes) => report.schema_sha256 = Some(format!("sha256:{}", hex_lower(&Sha256::digest(&schema_bytes)))),
            Err(error) => {
                eprintln!("cannot read schema {schema_path}: {error}");
                return ExitCode::from(2);
            }
        }
    }
    let status = report.status.clone();
    let output = match serde_json::to_value(report) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("cannot serialize verification report: {error}");
            return ExitCode::from(2);
        }
    };
    let code = if status == "pass" { 0 } else { 1 };
    emit(output, report_path, code)
}

fn emit(value: serde_json::Value, path: Option<String>, exit_code: u8) -> ExitCode {
    let serialized = match serde_json::to_string_pretty(&value) {
        Ok(text) => text + "\n",
        Err(error) => {
            eprintln!("cannot serialize JSON: {error}");
            return ExitCode::from(2);
        }
    };
    match path {
        Some(path) => {
            if let Err(error) = fs::write(&path, &serialized) {
                eprintln!("cannot write report {path}: {error}");
                return ExitCode::from(2);
            }
        }
        None => print!("{serialized}"),
    }
    if exit_code == 0 { ExitCode::SUCCESS } else { ExitCode::from(exit_code) }
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
