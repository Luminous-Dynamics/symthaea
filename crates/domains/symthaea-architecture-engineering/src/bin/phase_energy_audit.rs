use std::env;
use std::fs;
use std::process::ExitCode;

use symthaea_architecture_engineering::phase_energy_audit::{
    audit_application_phase_energy, ApplicationPhaseProtocolManifest,
    PhaseClockMappingManifest,
};
use symthaea_architecture_engineering::ppk2_trace_audit::Ppk2TraceAuditConfig;

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("phase-energy-audit: {message}");
    ExitCode::FAILURE
}

fn read_text(path: &std::ffi::OsStr, label: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("cannot read {label}: {error}"))
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(samples_path) = args.next() else {
        return fail(
            "usage: phase_energy_audit <samples.csv> <markers.csv> <outcomes.csv> <voltage_v> <clock-mapping.json> <phase-protocol.json> <receipt.json>",
        );
    };
    let Some(markers_path) = args.next() else {
        return fail("missing markers.csv argument");
    };
    let Some(outcomes_path) = args.next() else {
        return fail("missing outcomes.csv argument");
    };
    let Some(voltage_arg) = args.next() else {
        return fail("missing voltage_v argument");
    };
    let Some(mapping_path) = args.next() else {
        return fail("missing clock-mapping.json argument");
    };
    let Some(protocol_path) = args.next() else {
        return fail("missing phase-protocol.json argument");
    };
    let Some(receipt_path) = args.next() else {
        return fail("missing receipt.json output argument");
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

    let samples = match read_text(&samples_path, "samples CSV") {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let markers = match read_text(&markers_path, "markers CSV") {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let outcomes = match read_text(&outcomes_path, "application outcomes CSV") {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let mapping_json = match read_text(&mapping_path, "clock mapping JSON") {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let protocol_json = match read_text(&protocol_path, "phase protocol JSON") {
        Ok(value) => value,
        Err(error) => return fail(error),
    };

    let mapping: PhaseClockMappingManifest = match serde_json::from_str(&mapping_json) {
        Ok(value) => value,
        Err(error) => return fail(format!("invalid clock mapping JSON: {error}")),
    };
    let protocol: ApplicationPhaseProtocolManifest = match serde_json::from_str(&protocol_json) {
        Ok(value) => value,
        Err(error) => return fail(format!("invalid phase protocol JSON: {error}")),
    };
    let trace_config = Ppk2TraceAuditConfig {
        voltage_v,
        ..Ppk2TraceAuditConfig::default()
    };

    let audit = match audit_application_phase_energy(
        &samples,
        &markers,
        &outcomes,
        trace_config,
        &mapping,
        &protocol,
    ) {
        Ok(value) => value,
        Err(error) => return fail(error),
    };

    let receipt_json = match serde_json::to_vec_pretty(&audit.receipt) {
        Ok(value) => value,
        Err(error) => return fail(format!("cannot serialize phase receipt: {error}")),
    };
    if let Err(error) = fs::write(&receipt_path, receipt_json) {
        return fail(format!("cannot write phase receipt: {error}"));
    }

    println!(
        "PASS: {} events; {} phase-complete, {} with unattributed intervals, {} missing required phase, {} successful deadline-met events; mean energy/event {:?} uJ; receipt {}",
        audit.receipt.event_count,
        audit.receipt.complete_phase_coverage_events,
        audit.receipt.unattributed_interval_events,
        audit.receipt.missing_required_phase_events,
        audit.receipt.successful_completed_events,
        audit.receipt.mean_energy_per_successful_event_uj,
        receipt_path.to_string_lossy()
    );
    ExitCode::SUCCESS
}
