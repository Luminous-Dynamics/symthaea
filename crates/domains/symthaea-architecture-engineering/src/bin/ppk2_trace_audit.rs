use std::env;
use std::fs;
use std::process::ExitCode;

use symthaea_architecture_engineering::ppk2_trace_audit::{
    audit_ppk2_trace_csv, Ppk2TraceAuditConfig,
};

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("ppk2-trace-audit: {message}");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(samples_path) = args.next() else {
        return fail("usage: ppk2_trace_audit <samples.csv> <voltage_v> <receipt.json>");
    };
    let Some(voltage_arg) = args.next() else {
        return fail("usage: ppk2_trace_audit <samples.csv> <voltage_v> <receipt.json>");
    };
    let Some(receipt_path) = args.next() else {
        return fail("usage: ppk2_trace_audit <samples.csv> <voltage_v> <receipt.json>");
    };
    if args.next().is_some() {
        return fail("too many arguments");
    }

    let voltage_text = match voltage_arg.to_str() {
        Some(value) => value,
        None => return fail("voltage argument must be UTF-8"),
    };
    let voltage_v = match voltage_text.parse::<f64>() {
        Ok(value) if value.is_finite() && value > 0.0 => value,
        _ => return fail("voltage_v must be a finite positive number"),
    };

    let samples = match fs::read_to_string(&samples_path) {
        Ok(value) => value,
        Err(error) => return fail(format!("cannot read samples CSV: {error}")),
    };

    let config = Ppk2TraceAuditConfig {
        voltage_v,
        ..Ppk2TraceAuditConfig::default()
    };
    let audit = match audit_ppk2_trace_csv(&samples, config) {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let receipt = match serde_json::to_vec_pretty(&audit.receipt) {
        Ok(value) => value,
        Err(error) => return fail(format!("cannot serialize receipt: {error}")),
    };
    if let Err(error) = fs::write(&receipt_path, receipt) {
        return fail(format!("cannot write receipt: {error}"));
    }

    println!(
        "PASS: {} segments, {} samples, mean energy {:.6} uJ, receipt {}",
        audit.receipt.segment_count,
        audit.receipt.total_sample_count,
        audit.receipt.mean_segment_energy_uj,
        receipt_path.to_string_lossy()
    );
    ExitCode::SUCCESS
}
