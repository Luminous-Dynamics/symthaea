use std::env;
use std::fs;
use std::process::ExitCode;

use symthaea_architecture_engineering::nats_bench_mcu_audit::{
    audit_nats_bench_mcu_export, NATS_BENCH_MCU_V2_CONTRACT,
};

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("nats-bench-mcu-audit: {message}");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(hardware_path) = args.next() else {
        return fail(
            "usage: nats-bench-mcu-audit <hw_metrics.csv> <accuracies.csv> <failures.csv> [receipt.json]",
        );
    };
    let Some(accuracy_path) = args.next() else {
        return fail(
            "usage: nats-bench-mcu-audit <hw_metrics.csv> <accuracies.csv> <failures.csv> [receipt.json]",
        );
    };
    let Some(failures_path) = args.next() else {
        return fail(
            "usage: nats-bench-mcu-audit <hw_metrics.csv> <accuracies.csv> <failures.csv> [receipt.json]",
        );
    };
    let output_path = args.next();
    if args.next().is_some() {
        return fail("too many arguments");
    }

    let hardware = match fs::read_to_string(&hardware_path) {
        Ok(value) => value,
        Err(error) => return fail(format!("cannot read hardware metrics CSV: {error}")),
    };
    let accuracies = match fs::read_to_string(&accuracy_path) {
        Ok(value) => value,
        Err(error) => return fail(format!("cannot read accuracy CSV: {error}")),
    };
    let failures = match fs::read_to_string(&failures_path) {
        Ok(value) => value,
        Err(error) => return fail(format!("cannot read failures CSV: {error}")),
    };

    let audit = match audit_nats_bench_mcu_export(
        &hardware,
        &accuracies,
        &failures,
        NATS_BENCH_MCU_V2_CONTRACT,
    ) {
        Ok(value) => value,
        Err(error) => return fail(error),
    };

    let receipt = match serde_json::to_vec_pretty(&audit.receipt) {
        Ok(value) => value,
        Err(error) => return fail(format!("cannot serialize audit receipt: {error}")),
    };

    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, &receipt) {
            return fail(format!("cannot write audit receipt: {error}"));
        }
        println!(
            "PASS: {} measured + {} infeasible = {} candidates; receipt {}",
            audit.receipt.measured_candidates,
            audit.receipt.infeasible_candidates,
            audit.receipt.expected_total_candidates,
            path.to_string_lossy()
        );
    } else {
        println!("{}", String::from_utf8_lossy(&receipt));
    }

    ExitCode::SUCCESS
}
